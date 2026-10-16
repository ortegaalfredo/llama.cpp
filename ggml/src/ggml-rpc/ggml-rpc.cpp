#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "transport.h"
#include "log.h"
#include "rpc-compress.h"
#include "rpc-cache.h"

#include <array>
#include <chrono>
#include <cinttypes>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include <queue>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <atomic>

namespace fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(push, 1)
// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];

    int32_t use_count;
};

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_SET_TENSOR_2D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    RPC_CMD_CACHE_LIST,
    RPC_CMD_SET_TENSOR_CACHED,
    RPC_CMD_NONE,
    RPC_CMD_COUNT,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

static const char * rpc_cmd_name(enum rpc_cmd cmd) {
    switch (cmd) {
        case RPC_CMD_ALLOC_BUFFER:      return "ALLOC_BUFFER";
        case RPC_CMD_GET_ALIGNMENT:     return "GET_ALIGNMENT";
        case RPC_CMD_GET_MAX_SIZE:      return "GET_MAX_SIZE";
        case RPC_CMD_BUFFER_GET_BASE:   return "BUFFER_GET_BASE";
        case RPC_CMD_FREE_BUFFER:       return "FREE_BUFFER";
        case RPC_CMD_BUFFER_CLEAR:      return "BUFFER_CLEAR";
        case RPC_CMD_SET_TENSOR:        return "SET_TENSOR";
        case RPC_CMD_SET_TENSOR_HASH:   return "SET_TENSOR_HASH";
        case RPC_CMD_GET_TENSOR:        return "GET_TENSOR";
        case RPC_CMD_COPY_TENSOR:       return "COPY_TENSOR";
        case RPC_CMD_GRAPH_COMPUTE:     return "GRAPH_COMPUTE";
        case RPC_CMD_GET_DEVICE_MEMORY: return "GET_DEVICE_MEMORY";
        case RPC_CMD_INIT_TENSOR:       return "INIT_TENSOR";
        case RPC_CMD_GET_ALLOC_SIZE:    return "GET_ALLOC_SIZE";
        case RPC_CMD_HELLO:             return "HELLO";
        case RPC_CMD_DEVICE_COUNT:      return "DEVICE_COUNT";
        case RPC_CMD_GRAPH_RECOMPUTE:   return "GRAPH_RECOMPUTE";
        case RPC_CMD_MEMSET_TENSOR:     return "MEMSET_TENSOR";
        case RPC_CMD_SET_TENSOR_2D:     return "SET_TENSOR_2D";
        case RPC_CMD_GET_TENSOR_2D:     return "GET_TENSOR_2D";
        case RPC_CMD_COMM_INIT:         return "COMM_INIT";
        case RPC_CMD_COMM_ALLREDUCE:    return "COMM_ALLREDUCE";
        case RPC_CMD_COMM_FREE:         return "COMM_FREE";
        case RPC_CMD_CACHE_LIST:        return "CACHE_LIST";
        case RPC_CMD_SET_TENSOR_CACHED: return "SET_TENSOR_CACHED";
        case RPC_CMD_NONE:              return "NONE";
        default:                        return "UNKNOWN";
    }
}

// weight tensors at least this big go through the server's disk cache (see rpc-cache.h)
const size_t RPC_CACHE_MIN_SIZE = 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
const size_t GRAPH_CACHE_MAX = 1024;

struct rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_device_count_rsp {
    uint32_t device_count;
};

struct rpc_msg_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_req {
    uint32_t device;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_req {
    uint32_t device;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

struct rpc_msg_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

struct rpc_msg_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

struct rpc_msg_set_tensor_hash_rsp {
    uint8_t result;
};

// sets a tensor from the server's disk cache entry with this digest
struct rpc_msg_set_tensor_cached_req {
    rpc_tensor tensor;
    uint64_t   offset;
    uint64_t   size;
    uint8_t    digest[32];
};

struct rpc_msg_set_tensor_cached_rsp {
    uint8_t result;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_get_device_memory_req {
    uint32_t device;
};

struct rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rpc_msg_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

struct rpc_msg_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

struct rpc_msg_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t port;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    char     host[64];  // rank > 0: rank 0's host
};

struct rpc_msg_comm_init_rsp {
    uint8_t ok;
};

struct rpc_msg_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

struct rpc_msg_comm_free_req {
    uint32_t device;
};

#pragma pack(pop)

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

class rpc_dispatcher;
struct ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    void                            * base_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    const uint64_t fnv_prime = 0x100000001b3ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static bool send_msg(socket_ptr sock, const void * msg, size_t msg_size) {
    if (!sock->send_data(&msg_size, sizeof(msg_size))) {
        return false;
    }
    if (!sock->send_data(msg, msg_size)) {
        return false;
    }
    return sock->flush();
}

static bool recv_msg(socket_ptr sock, void * msg, size_t msg_size) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return sock->recv_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    } catch (const std::bad_alloc & e) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    return sock->recv_data(input.data(), size);
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    try {
        port = std::stoi(endpoint.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    uint8_t cmd_byte = cmd;
    if (!sock->send_data(&cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    if (!sock->send_data(&input_size, sizeof(input_size))) {
        return false;
    }
    if (!sock->send_data(input, input_size)) {
        return false;
    }
    return sock->flush();
}

// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool recv_rpc_response(socket_ptr sock, void * output, size_t output_size) {
    uint64_t out_size;
    if (!sock->recv_data(&out_size, sizeof(out_size))) {
        return false;
    }
    if (out_size != output_size) {
        return false;
    }
    return sock->recv_data(output, output_size);
}

// Same as above, but the response size is not known a priori (the payload may
// be compressed), so it is received into a vector sized from the wire.
static bool recv_rpc_response_vector(socket_ptr sock, std::vector<uint8_t> & output) {
    uint64_t out_size;
    if (!sock->recv_data(&out_size, sizeof(out_size))) {
        return false;
    }
    if (out_size > MAX_CHUNK_SIZE) {
        return false;
    }
    output.resize(out_size);
    return sock->recv_data(output.data(), out_size);
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    return send_rpc_cmd(sock, cmd, input, input_size) && recv_rpc_response(sock, output, output_size);
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
static constexpr auto RPC_BUSY_SPIN_IDLE_TIME = std::chrono::milliseconds(100);

static inline void rpc_cpu_relax() {
#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

// conn_caps layout used by the compression negotiation. The RDMA transport
// consumes the whole caps array (rdma_caps is exactly RPC_CONN_CAPS_SIZE
// bytes: qpn, psn, gid), so the compression level only gets a byte of its own
// when the RDMA transport is not compiled in. With RDMA, compression stays
// disabled: RDMA is for local networks, where compression buys nothing.
#ifdef GGML_RPC_RDMA
static constexpr size_t RPC_CAPS_COMP_OFF = SIZE_MAX;   // no byte available
#else
static constexpr size_t RPC_CAPS_COMP_OFF = RPC_CONN_CAPS_SIZE - 1;
#endif

// a second caps byte carries feature flags (zero from peers that predate it)
#ifdef GGML_RPC_RDMA
static constexpr size_t RPC_CAPS_FEAT_OFF = SIZE_MAX;
#else
static constexpr size_t RPC_CAPS_FEAT_OFF = RPC_CONN_CAPS_SIZE - 2;
#endif
// the server has a disk cache and serves CACHE_LIST / SET_TENSOR_CACHED
static constexpr uint8_t RPC_FEAT_CACHE = 0x01;

static void rpc_caps_set_features(uint8_t * caps, uint8_t features) {
    if (RPC_CAPS_FEAT_OFF != SIZE_MAX) {
        caps[RPC_CAPS_FEAT_OFF] = features;
    }
}

static uint8_t rpc_caps_get_features(const uint8_t * caps) {
    return RPC_CAPS_FEAT_OFF != SIZE_MAX ? caps[RPC_CAPS_FEAT_OFF] : 0;
}

// the caps byte carries the codec set version in the high nibble and the
// level in the low nibble, so a peer with a different codec set (including
// earlier builds of this feature) negotiates RPC_COMP_NONE instead of
// misreading payloads
static void rpc_caps_set_comp(uint8_t * caps, uint8_t level) {
    if (RPC_CAPS_COMP_OFF == SIZE_MAX) {
        return;
    }
    caps[RPC_CAPS_COMP_OFF] = (uint8_t) ((RPC_COMP_CAPS_VERSION << 4) | (level & 0xf));
}

static uint8_t rpc_caps_get_comp(const uint8_t * caps) {
    if (RPC_CAPS_COMP_OFF == SIZE_MAX) {
        return RPC_COMP_NONE;
    }
    const uint8_t v = caps[RPC_CAPS_COMP_OFF];
    if ((v >> 4) != RPC_COMP_CAPS_VERSION) {
        return RPC_COMP_NONE;
    }
    return v & 0xf;
}

// the level both sides must use: the more conservative of the two, so a
// lossy level never gets applied unless both peers accept it
static uint8_t rpc_comp_negotiate(uint8_t local, uint8_t remote) {
    if (!rpc_compress_level_valid(local) || !rpc_compress_level_valid(remote)) {
        return RPC_COMP_NONE;
    }
    return local < remote ? local : remote;
}

// The most aggressive level allowed for a payload of `size` bytes of
// `tensor`, given the level negotiated for the connection. Lossy levels are
// only allowed for activations:
//  - float tensors in a compute buffer (never weights, never the KV cache,
//    which lives in a buffer with usage ANY)
//  - rows at least RPC_COMP_MIN_ROW wide, and a payload of whole rows
//  - for reads, not a graph output: everything the application reads back
//    after a compute (logits, embeddings, ...) must be flagged as an output
//    so the allocator does not reuse its memory, while the scheduler's own
//    copies between backends read intermediate tensors that are not
// Everything else is sent losslessly.
static uint8_t rpc_payload_level(uint8_t negotiated, const ggml_tensor * tensor, size_t size, bool is_read) {
    if (!rpc_compress_level_lossy(negotiated)) {
        return negotiated;
    }
    const bool lossy_ok =
        tensor->buffer != nullptr &&
        ggml_backend_buffer_get_usage(tensor->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
        (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) &&
        tensor->ne[0] >= RPC_COMP_MIN_ROW &&
        size % ggml_row_size(tensor->type, tensor->ne[0]) == 0 &&
        !(is_read && (tensor->flags & GGML_TENSOR_FLAG_OUTPUT));
    return lossy_ok ? negotiated : (uint8_t) RPC_COMP_ZSTD;
}

// row length handed to the lossy codecs
static uint32_t rpc_payload_row_len(const ggml_tensor * tensor) {
    return tensor->ne[0] <= UINT32_MAX ? (uint32_t) tensor->ne[0] : 0;
}

// input serialization format: | rpc_tensor | flags (1 byte) | offset (8 bytes) | data (size bytes)
// bit 0 of the flags byte is the hash-cache flag, bit 1 marks a compressed
// payload (see rpc-compress.h for the wire format of the data section).
static constexpr uint8_t RPC_SET_TENSOR_FLAG_CACHE     = 0x01;
static constexpr uint8_t RPC_SET_TENSOR_FLAG_COMPRESSED = 0x02;

// When compression is negotiated, GET_TENSOR / GET_TENSOR_2D requests carry
// the most aggressive level the client allows for that payload: the server
// cannot tell activations from weights or outputs on its own.
struct rpc_msg_get_tensor_comp_req {
    rpc_msg_get_tensor_req req;
    uint64_t               level;
};

struct rpc_msg_get_tensor_2d_comp_req {
    rpc_msg_get_tensor_2d_req req;
    uint64_t                  level;
};

// Unpacks a GET_TENSOR / GET_TENSOR_2D response received with compression
// negotiated: | flags (1 byte) | data |, where data is compressed when the
// compressed bit is set and raw otherwise.
static void rpc_unpack_get_response(const std::vector<uint8_t> & wire, void * out, size_t out_size,
                                    enum ggml_type type, enum rpc_cmd cmd) {
    if (wire.empty()) {
        GGML_ABORT("empty %s response\n", rpc_cmd_name(cmd));
    }
    if (wire[0] & RPC_SET_TENSOR_FLAG_COMPRESSED) {
        if (!rpc_decompress_payload(out, out_size, wire.data() + 1, wire.size() - 1, type)) {
            GGML_ABORT("failed to decompress %s response\n", rpc_cmd_name(cmd));
        }
        LOG_DBG3("[%s] %s decompressed %s -> %s\n", __func__, rpc_cmd_name(cmd),
                 rpc_log_format_size(wire.size() - 1).c_str(), rpc_log_format_size(out_size).c_str());
    } else {
        if (wire.size() != 1 + out_size) {
            GGML_ABORT("unexpected %s response size: %zu (expected %zu)\n", rpc_cmd_name(cmd), wire.size(), 1 + out_size);
        }
        memcpy(out, wire.data() + 1, out_size);
    }
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
// `comp_level` receives the compression level negotiated with the server and
// `features` the server's feature flags.
static bool negotiate_hello(const std::shared_ptr<socket_t> & sock, uint8_t & comp_level, uint8_t & features) {
    rpc_msg_hello_req request = {};
    rpc_msg_hello_rsp response = {};

    sock->get_caps(request.conn_caps);
    const uint8_t local_comp = rpc_compress_env_level();
    rpc_caps_set_comp(request.conn_caps, local_comp);

    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);

    if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        return false;
    }

    sock->update_caps(response.conn_caps);

    // compression is negotiated down to the level both sides support
    comp_level = rpc_comp_negotiate(local_comp, rpc_caps_get_comp(response.conn_caps));
    features   = rpc_caps_get_features(response.conn_caps);
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, transport: %s, compression: %s\n",
            __func__, response.major, response.minor, response.patch, sock->transport_name(),
            rpc_compress_level_name(comp_level));
    return true;
}

template <typename T>
class message_queue {
public:
    message_queue() {}

    bool push(const T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        if (interrupted) {
            return false;
        }
        queue.push(value);
        cvar.notify_all();
        return true;
    }

    bool pop(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.wait(lock, [this] { return !queue.empty() || interrupted; });
        if (interrupted) {
            return false;
        }
        *out = queue.front();
        queue.pop();
        return true;
    }

    bool try_pop(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        if (interrupted || queue.empty()) {
            return false;
        }
        *out = queue.front();
        queue.pop();
        return true;
    }

    void interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        interrupted = true;
        lock.unlock();
        cvar.notify_all();
    }

private:
    bool interrupted = false;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

class rpc_dispatcher {
public:
    rpc_dispatcher() {
    }

    // compression level negotiated with the server in the HELLO handshake.
    // RPC_COMP_NONE means every payload goes over the wire unchanged.
    uint8_t comp_level = RPC_COMP_NONE;
    // the server's feature flags (RPC_FEAT_*)
    uint8_t features = 0;

    // whether the server's disk cache holds an entry with this digest. The
    // list of entries is fetched once, on first use, so a model load costs a
    // single round trip for the cache however many tensors it has.
    bool cache_contains(const rpc_digest & digest);
    // records an entry the server is about to store
    void cache_add(const rpc_digest & digest);
    // async send whose response is a single status byte, checked by the
    // dispatcher: a failure aborts
    void send_async_checked(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size);

    void send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size);
    void send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size);
    void send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size);
    void send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size);
    // blocking send whose response is received into a vector; used for
    // GET_TENSOR responses, which may be compressed and therefore have an
    // unknown wire size
    void send_recv_vector(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, std::vector<uint8_t> & output);
    // async variant: the response is received into a buffer owned by the
    // message and decompressed into `output` by the dispatcher thread
    void send_async_recv_decompressed(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size,
                                      void * output, size_t output_size, enum ggml_type tensor_type);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    void event_free(ggml_backend_event_t event);
    void event_synchronize(ggml_backend_event_t event);
    void event_record(ggml_backend_event_t event);
    void synchronize();
    void busy_spin_acquire();
    void busy_spin_release();
    void graph_compute(uint32_t device, const ggml_cgraph * cgraph);

    void start(const std::string & endpoint);
    void work();
    void receive_work();

    ~rpc_dispatcher();

private:
    struct rpc_msg {
        rpc_cmd                       cmd;
        std::shared_ptr<const void>   input;
        size_t                        input_size;
        void                        * output;
        size_t                        output_size;
        // when set, the response is received into this vector instead of
        // `output` (used for possibly-compressed GET_TENSOR responses)
        std::vector<uint8_t>        * recv_vector = nullptr;
        // async variant: the response is received into a buffer owned by the
        // message and then decompressed into `output` by the dispatcher
        // thread, before the completion event fires
        std::vector<uint8_t>          recv_buf = {};
        enum ggml_type                tensor_type = GGML_TYPE_F32;
        bool                          decompress_into_output = false;
        std::promise<void>            completion;
        std::chrono::steady_clock::time_point t_sent = {};
        // the response is one status byte that must be non-zero
        bool                          checked = false;
        uint8_t                       status = 0;

        bool expects_response() const {
            return recv_vector != nullptr || decompress_into_output || output != nullptr || checked;
        }
    };
    using rpc_msg_ptr   = std::shared_ptr<rpc_msg>;
    using rpc_msg_queue = message_queue<rpc_msg_ptr>;
    struct rpc_event {
        rpc_msg_ptr              msg;
        std::shared_future<void> sf;
    };
    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;
    // receives the response of msg (if it has one) and completes it
    void finish(const rpc_msg_ptr & msg);
    // waits for the next message, spinning while busy spinning is requested
    // and commands keep coming
    bool next(rpc_msg_queue & q, rpc_msg_ptr & msg, std::chrono::steady_clock::time_point & last);

    rpc_msg_queue    queue;
    socket_ptr       sock;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
    // Request pipelining (TCP only). The worker thread sends each request as
    // soon as it is queued and hands the message to `inflight`; the receiver
    // thread reads the responses in the same order and completes the
    // messages. Several reads queued back to back (e.g. the outputs of
    // backend sampling) then cost one network round trip instead of one each.
    // The server handles commands strictly in order, so responses arrive in
    // request order. Without pipelining the worker does both, one command at
    // a time.
    bool             pipelined = false;
    rpc_msg_queue    inflight;
    std::thread      recv_thread;

    std::mutex       cache_mutex;
    bool             cache_listed = false;
    std::unordered_set<rpc_digest, rpc_digest_hash> cache_entries;
};


static void rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

void rpc_dispatcher::send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = nullptr;
    msg->output_size = 0;
    GGML_ASSERT(queue.push(msg));
    auto future = msg->completion.get_future();
    future.wait();
}

void rpc_dispatcher::send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = nullptr;
    msg->output_size = 0;
    GGML_ASSERT(queue.push(msg));
}

void rpc_dispatcher::send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = output;
    msg->output_size = output_size;
    GGML_ASSERT(queue.push(msg));
    auto future = msg->completion.get_future();
    future.wait();
}

void rpc_dispatcher::send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = output;
    msg->output_size = output_size;
    GGML_ASSERT(queue.push(msg));
}

// Blocking send whose response is received into a vector. Used for GET_TENSOR
// responses, which may be compressed and therefore have an unknown wire size.
void rpc_dispatcher::send_recv_vector(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size,
                                      std::vector<uint8_t> & output) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = nullptr;
    msg->output_size = 0;
    msg->recv_vector = &output;
    GGML_ASSERT(queue.push(msg));
    auto future = msg->completion.get_future();
    future.wait();
}

// Async variant of the above: the response is received into a buffer owned by
// the message, then decompressed into `output` by the dispatcher thread. The
// caller's buffer is only touched after the decompression succeeded, and the
// completion event fires after that, so the usual event-synchronize
// discipline is enough to make the write visible.
void rpc_dispatcher::send_async_recv_decompressed(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size,
                                                  void * output, size_t output_size, enum ggml_type tensor_type) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = output;
    msg->output_size = output_size;
    msg->tensor_type = tensor_type;
    msg->decompress_into_output = true;
    GGML_ASSERT(queue.push(msg));
}

void rpc_dispatcher::send_async_checked(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = nullptr;
    msg->output_size = 0;
    msg->checked = true;
    GGML_ASSERT(queue.push(msg));
}

bool rpc_dispatcher::cache_contains(const rpc_digest & digest) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (!cache_listed) {
        std::vector<uint8_t> list;
        send_recv_vector(RPC_CMD_CACHE_LIST, nullptr, 0, list);
        if (list.size() % sizeof(digest.b) != 0) {
            GGML_ABORT("unexpected CACHE_LIST response size: %zu\n", list.size());
        }
        for (size_t i = 0; i < list.size(); i += sizeof(digest.b)) {
            rpc_digest d;
            memcpy(d.b.data(), list.data() + i, sizeof(d.b));
            cache_entries.insert(d);
        }
        cache_listed = true;
        LOG_DBG("[%s] server cache holds %zu entries\n", __func__, cache_entries.size());
    }
    return cache_entries.count(digest) > 0;
}

void rpc_dispatcher::cache_add(const rpc_digest & digest) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache_entries.insert(digest);
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = new rpc_event;
    ev->msg = std::make_shared<rpc_msg>();
    ev->msg->cmd = RPC_CMD_NONE;
    ev->sf = ev->msg->completion.get_future().share();
    GGML_ASSERT(queue.push(ev->msg));
    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

void rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    delete ev;
}

void rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

void rpc_dispatcher::event_record(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->msg = std::make_shared<rpc_msg>();
    ev->msg->cmd = RPC_CMD_NONE;
    ev->sf = ev->msg->completion.get_future().share();
    GGML_ASSERT(queue.push(ev->msg));
}

void rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.push(msg));
    msg->completion.get_future().wait();
}

void rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

void rpc_dispatcher::busy_spin_release() {
    const unsigned previous = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(previous > 0);
}

void rpc_dispatcher::start(const std::string & endpoint) {
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    if (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), port);
    if (sock == nullptr) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    if (!negotiate_hello(sock, comp_level, features)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    // the TCP transport sends and receives with plain send()/recv() on the
    // socket, which may run concurrently from two threads; RDMA keeps the
    // single-threaded path
    pipelined = strcmp(sock->transport_name(), "TCP") == 0 && std::getenv("GGML_RPC_NO_PIPELINE") == nullptr;
    LOG_DBG("[%s] connected to %s, pipelining: %s\n", __func__, endpoint.c_str(), pipelined ? "on" : "off");
    running = true;
    thread = std::thread(rpc_dispatcher_trampoline, this);
    if (pipelined) {
        recv_thread = std::thread([this] { receive_work(); });
    }
}

bool rpc_dispatcher::next(rpc_msg_queue & q, rpc_msg_ptr & msg, std::chrono::steady_clock::time_point & last) {
    while (running) {
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        const bool spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - last < RPC_BUSY_SPIN_IDLE_TIME;
        if (!spin) {
            return q.pop(&msg);
        }
        if (q.try_pop(&msg)) {
            return true;
        }
        rpc_cpu_relax();
    }
    return false;
}

void rpc_dispatcher::finish(const rpc_msg_ptr & msg) {
    if (msg->cmd != RPC_CMD_NONE && msg->expects_response()) {
        bool status;
        if (msg->recv_vector) {
            status = recv_rpc_response_vector(sock, *msg->recv_vector);
        } else if (msg->decompress_into_output) {
            status = recv_rpc_response_vector(sock, msg->recv_buf);
            if (status) {
                // the server prefixes a flags byte when compression was
                // negotiated; unpack the payload into the caller's buffer
                rpc_unpack_get_response(msg->recv_buf, msg->output, msg->output_size, msg->tensor_type, msg->cmd);
                msg->recv_buf = {};
            }
        } else if (msg->checked) {
            status = recv_rpc_response(sock, &msg->status, sizeof(msg->status));
            if (status && msg->status == 0) {
                GGML_ABORT("%s failed on the server\n", rpc_cmd_name(msg->cmd));
            }
        } else {
            status = recv_rpc_response(sock, msg->output, msg->output_size);
        }
        RPC_STATUS_ASSERT(status);
    }
    if (msg->cmd != RPC_CMD_NONE && rpc_log_debug_level() >= 3) {
        const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - msg->t_sent;
        LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(msg->cmd), elapsed.count());
    } else if (msg->cmd == RPC_CMD_NONE) {
        LOG_DBG3("[%s] sync barrier\n", __func__);
    }
    msg->completion.set_value();
}

void rpc_dispatcher::work() {
    auto last_cmd = std::chrono::steady_clock::now();
    rpc_msg_ptr msg_ptr;
    while (next(queue, msg_ptr, last_cmd)) {
        if (msg_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(msg_ptr->cmd),
                     rpc_log_format_size(msg_ptr->input_size).c_str(), rpc_log_format_size(msg_ptr->output_size).c_str());
            msg_ptr->t_sent = std::chrono::steady_clock::now();
            bool status = send_rpc_cmd(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size);
            RPC_STATUS_ASSERT(status);
        }
        if (pipelined && msg_ptr->cmd != RPC_CMD_NONE && !msg_ptr->expects_response()) {
            // nothing to receive: done once sent, so a blocking send does not
            // wait for the responses of earlier requests
            msg_ptr->completion.set_value();
        } else if (pipelined) {
            // barriers go through the receiver too, so they complete only
            // after every earlier response has arrived
            GGML_ASSERT(inflight.push(msg_ptr));
        } else {
            finish(msg_ptr);
        }
        msg_ptr = nullptr;
        last_cmd = std::chrono::steady_clock::now();
    }
}

void rpc_dispatcher::receive_work() {
    auto last_cmd = std::chrono::steady_clock::now();
    rpc_msg_ptr msg_ptr;
    while (next(inflight, msg_ptr, last_cmd)) {
        finish(msg_ptr);
        msg_ptr = nullptr;
        last_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = false;
    queue.interrupt();
    inflight.interrupt();
    if (thread.joinable()) {
        thread.join();
    }
    if (recv_thread.joinable()) {
        recv_thread.join();
    }
    sock = nullptr;
}

static std::shared_ptr<rpc_dispatcher> get_dispatcher(const std::string & endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    auto it = dispatchers.find(endpoint);
    if (it != dispatchers.end()) {
        if (auto dispatcher = it->second.lock()) {
            return dispatcher;
        }
    }

    auto dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    return dispatcher;
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_msg_free_buffer_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_BUFFER, request, sizeof(*request));
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    auto request = std::make_shared<rpc_msg_buffer_get_base_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_msg_buffer_get_base_rsp response;
    ctx->dispatcher->send(RPC_CMD_BUFFER_GET_BASE, request, sizeof(*request), &response, sizeof(response));
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor, const rpc_dispatcher * dispatcher = nullptr) {
    rpc_tensor result;
    if (!tensor) {
        memset(&result, 0, sizeof(result));
        return result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        if (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher.get() == dispatcher)) {
            result.buffer = ctx->remote_ptr;
            result.data = reinterpret_cast<uint64_t>(tensor->data);
        } else {
            result.buffer = 0;
            result.data = 0;
        }
    } else {
        result.buffer = 0;
        result.data   = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, sizeof(result.name));
    result.use_count = 0;

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

static enum ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        auto request = std::make_shared<rpc_msg_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, sizeof(*request));
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_msg_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, sizeof(*request));
}

static std::shared_ptr<uint8_t> serialize_set_tensor(const rpc_tensor & rpc_tensor, uint8_t flags, uint64_t offset, const void * data, size_t size, size_t & input_size) {
    input_size = sizeof(rpc_tensor) + sizeof(flags) + sizeof(offset) + size;
    uint8_t * input = new uint8_t[input_size]();
    uint8_t * p = input;
    memcpy(p, &rpc_tensor, sizeof(rpc_tensor)); p += sizeof(rpc_tensor);
    memcpy(p, &flags, sizeof(flags)); p += sizeof(flags);
    memcpy(p, &offset, sizeof(offset));     p += sizeof(offset);
    memcpy(p, data, size);
    return std::shared_ptr<uint8_t>(input, std::default_delete<uint8_t[]>());
}

// Same as serialize_set_tensor, but the data section is compressed at up to
// `level` (see rpc_payload_level) when that makes it smaller; the compressed
// bit of the flags byte tells the server which form it gets.
static std::shared_ptr<uint8_t> serialize_set_tensor_compressed(const ggml_tensor * tensor, const rpc_tensor & rpc_tensor,
                                                                uint8_t flags, uint64_t offset, const void * data, size_t size,
                                                                uint8_t level, size_t & input_size) {
    std::vector<uint8_t> wire;
    const uint8_t codec = rpc_compress_payload(wire, data, size, tensor->type, rpc_payload_row_len(tensor), level);
    if (codec != RPC_CODEC_RAW) {
        flags |= RPC_SET_TENSOR_FLAG_COMPRESSED;
        LOG_DBG3("[%s] tensor %s compressed %s -> %s (codec %u)\n", __func__, rpc_tensor.name,
                 rpc_log_format_size(size).c_str(), rpc_log_format_size(wire.size()).c_str(), codec);
        return serialize_set_tensor(rpc_tensor, flags, offset, wire.data(), wire.size(), input_size);
    }
    // incompressible or compression disabled: send raw
    return serialize_set_tensor(rpc_tensor, flags, offset, data, size, input_size);
}


// Weight upload through the server's disk cache. Returns true when the tensor
// was set from a cache entry (nothing else to send); otherwise adds the cache
// flag to `flags` when the server should store the data it is about to get.
// Only weights take this path: compute-buffer inputs (the activations the
// scheduler copies between backends) change every ubatch.
// When `digest_out` is not null it receives the digest, if one was computed.
static bool rpc_tensor_cacheable(const std::shared_ptr<rpc_dispatcher> & dispatcher, const ggml_tensor * tensor, size_t size) {
    return (dispatcher->features & RPC_FEAT_CACHE) && size >= RPC_CACHE_MIN_SIZE &&
           tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static void rpc_send_set_tensor_cached(const std::shared_ptr<rpc_dispatcher> & dispatcher, const rpc_tensor & rpc_tensor,
                                       size_t offset, size_t size, const rpc_digest & digest) {
    auto request = std::make_shared<rpc_msg_set_tensor_cached_req>();
    request->tensor = rpc_tensor;
    request->offset = offset;
    request->size   = size;
    memcpy(request->digest, digest.b.data(), sizeof(request->digest));
    // pipelined: the status is checked by the dispatcher
    dispatcher->send_async_checked(RPC_CMD_SET_TENSOR_CACHED, request, sizeof(*request));
}

static bool rpc_set_tensor_from_cache(const std::shared_ptr<rpc_dispatcher> & dispatcher, const ggml_tensor * tensor,
                                      const rpc_tensor & rpc_tensor, size_t offset, const void * data, size_t size,
                                      uint8_t & flags, rpc_digest * digest_out = nullptr) {
    if (!rpc_tensor_cacheable(dispatcher, tensor, size)) {
        return false;
    }
    const rpc_digest digest = rpc_cache_digest(data, size);
    if (digest_out != nullptr) {
        *digest_out = digest;
    }
    if (dispatcher->cache_contains(digest)) {
        rpc_send_set_tensor_cached(dispatcher, rpc_tensor, offset, size, digest);
        LOG_DBG2("[%s] tensor: %s, cache hit %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), digest.hex().c_str());
        return true;
    }
    dispatcher->cache_add(digest);
    flags |= RPC_SET_TENSOR_FLAG_CACHE;
    return false;
}

static void rpc_buffer_set_tensor_impl(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset,
                                       size_t size, rpc_digest * digest_out) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t flags = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    if (rpc_set_tensor_from_cache(ctx->dispatcher, tensor, rpc_tensor, offset, data, size, flags, digest_out)) {
        return;
    }
    size_t input_size;
    const uint8_t level = rpc_payload_level(ctx->dispatcher->comp_level, tensor, size, false);
    auto input = serialize_set_tensor_compressed(tensor, rpc_tensor, flags, offset, data, size, level, input_size);
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR, input, input_size);
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    rpc_buffer_set_tensor_impl(buffer, tensor, data, offset, size, nullptr);
}

static void ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8) | size (8) | n_copies (8) | stride (8) | data |
    // When compression is negotiated, a flags byte is inserted after the
    // header and the data section may be compressed (see rpc-compress.h).
    // Peers that negotiated RPC_COMP_NONE use the original format, so mixed
    // versions stay compatible.
    const uint64_t header[4] = { offset, size, n_copies, stride_tensor };
    const size_t header_size = sizeof(rpc_tensor) + 4*sizeof(uint64_t);

    // gather the strided copies into one contiguous buffer first
    std::vector<uint8_t> gathered(size*n_copies);
    for (size_t i = 0; i < n_copies; i++) {
        memcpy(gathered.data() + i*size, (const char *)data + i*stride_data, size);
    }

    const bool compress = ctx->dispatcher->comp_level != RPC_COMP_NONE;
    std::vector<uint8_t> wire;
    uint8_t flags = 0;
    if (compress) {
        // every copy must be whole rows for the lossy codecs: checking the
        // per-copy size covers the gathered total too
        const uint8_t level = rpc_payload_level(ctx->dispatcher->comp_level, tensor, size, false);
        const uint8_t codec = rpc_compress_payload(wire, gathered.data(), gathered.size(), tensor->type,
                                                   rpc_payload_row_len(tensor), level);
        if (codec != RPC_CODEC_RAW) {
            flags = RPC_SET_TENSOR_FLAG_COMPRESSED;
            LOG_DBG3("[%s] tensor %s compressed %s -> %s (codec %u)\n", __func__, rpc_tensor.name,
                     rpc_log_format_size(gathered.size()).c_str(), rpc_log_format_size(wire.size()).c_str(), codec);
        }
        // incompressible: fall through and send the gathered bytes raw
    }
    const void * payload = (flags & RPC_SET_TENSOR_FLAG_COMPRESSED) ? (const void *)wire.data() : (const void *)gathered.data();
    const size_t payload_size = (flags & RPC_SET_TENSOR_FLAG_COMPRESSED) ? wire.size() : gathered.size();

    size_t input_size = header_size + (compress ? 1 : 0) + payload_size;
    uint8_t * input = new uint8_t[input_size]();
    uint8_t * dest = input;
    memcpy(dest, &rpc_tensor, sizeof(rpc_tensor));
    dest += sizeof(rpc_tensor);
    memcpy(dest, header, sizeof(header));
    dest += sizeof(header);
    if (compress) {
        *dest = flags;
        dest += 1;
    }
    memcpy(dest, payload, payload_size);
    std::shared_ptr<uint8_t> input_ptr(input, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR_2D, input_ptr, input_size);
}

static void ggml_backend_rpc_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    auto request = std::make_shared<rpc_msg_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->offset   = offset;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    const size_t total = size*n_copies;
    if (ctx->dispatcher->comp_level == RPC_COMP_NONE) {
        if (stride_data == size) {
            ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, sizeof(*request), data, total);
        } else {
            std::vector<uint8_t> packed(total);
            ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, sizeof(*request), packed.data(), packed.size());
            for (size_t i = 0; i < n_copies; i++) {
                memcpy((char *)data + i*stride_data, packed.data() + i*size, size);
            }
        }
        return;
    }
    // the response may be compressed, so its wire size is unknown: receive
    // into a scratch vector, decompress, then scatter the copies
    auto comp_request = std::make_shared<rpc_msg_get_tensor_2d_comp_req>();
    comp_request->req   = *request;
    comp_request->level = rpc_payload_level(ctx->dispatcher->comp_level, tensor, size, true);
    std::vector<uint8_t> wire;
    ctx->dispatcher->send_recv_vector(RPC_CMD_GET_TENSOR_2D, comp_request, sizeof(*comp_request), wire);
    std::vector<uint8_t> plain(total);
    rpc_unpack_get_response(wire, plain.data(), total, tensor->type, RPC_CMD_GET_TENSOR_2D);
    if (stride_data == size) {
        memcpy(data, plain.data(), total);
    } else {
        for (size_t i = 0; i < n_copies; i++) {
            memcpy((char *)data + i*stride_data, plain.data() + i*size, size);
        }
    }
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_msg_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    if (ctx->dispatcher->comp_level == RPC_COMP_NONE) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, sizeof(*request), data, size);
        return;
    }
    // the response may be compressed, so its wire size is unknown: receive
    // into a scratch vector, then decompress into the caller's buffer
    auto comp_request = std::make_shared<rpc_msg_get_tensor_comp_req>();
    comp_request->req   = *request;
    comp_request->level = rpc_payload_level(ctx->dispatcher->comp_level, tensor, size, true);
    std::vector<uint8_t> wire;
    ctx->dispatcher->send_recv_vector(RPC_CMD_GET_TENSOR, comp_request, sizeof(*comp_request), wire);
    rpc_unpack_get_response(wire, data, size, tensor->type, RPC_CMD_GET_TENSOR);
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        if (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            return false;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        auto request = std::make_shared<rpc_msg_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_msg_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, sizeof(*request), &response, sizeof(response));
        return response.result;
    }
    return false;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_msg_buffer_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_BUFFER_CLEAR, request, sizeof(*request));
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_rpc_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    auto request = std::make_shared<rpc_msg_alloc_buffer_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_msg_alloc_buffer_rsp response;

    auto dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_BUFFER, request, sizeof(*request), &response, sizeof(response));
    if (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str(),
                response.remote_ptr, rpc_log_format_size(response.remote_size).c_str());
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{dispatcher, nullptr, response.remote_ptr},
            response.remote_size);
        return buffer;
    } else {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str());
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_msg_get_alignment_req>();
    request->device = device;
    rpc_msg_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, sizeof(*request), &response, sizeof(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_size(response.alignment).c_str());
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_msg_get_max_size_req>();
    request->device = device;
    rpc_msg_get_max_size_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_SIZE, request, sizeof(*request), &response, sizeof(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_size(response.max_size).c_str());
    return response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_size_may_expand(tensor->op);

    if (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        const size_t min_size = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        struct alloc_size_cache_key {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_size_cache_key key = {};
        key.device = buft_ctx->device;
        key.type = tensor->type;
        key.op = tensor->op;
        memcpy(key.op_params, tensor->op_params, sizeof(key.op_params));
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            key.ne[i] = (uint32_t)tensor->ne[i];
            key.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((const uint8_t *)&key, sizeof(key));
        cache_hash = fnv_hash((const uint8_t *)buft_ctx->endpoint.data(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        static std::mutex cache_mutex;
        static std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(cache_hash);
            if (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_size(it->second).c_str());
                return std::max<size_t>(it->second, min_size);
            }
        }

        auto request = std::make_shared<rpc_msg_get_alloc_size_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_msg_get_alloc_size_rsp response;
        auto dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_SIZE, request, sizeof(*request), &response, sizeof(response));

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_hash] = response.alloc_size;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_size(response.alloc_size).c_str());

        return std::max<size_t>(response.alloc_size, min_size);
    }

    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    if (rpc_set_tensor_from_cache(ctx->dispatcher, tensor, rpc_tensor, offset, data, size, cache_flag)) {
        return;
    }
    size_t input_size;
    const uint8_t level = rpc_payload_level(ctx->dispatcher->comp_level, tensor, size, false);
    auto input = serialize_set_tensor_compressed(tensor, rpc_tensor, cache_flag, offset, data, size, level, input_size);
    ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR, input, input_size);
}

static void ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_msg_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    if (ctx->dispatcher->comp_level == RPC_COMP_NONE) {
        ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, sizeof(*request), data, size);
        return;
    }
    // the response may be compressed, so its wire size is unknown: receive
    // into a buffer owned by the message, then decompress into `data`
    auto comp_request = std::make_shared<rpc_msg_get_tensor_comp_req>();
    comp_request->req   = *request;
    comp_request->level = rpc_payload_level(ctx->dispatcher->comp_level, tensor, size, true);
    ctx->dispatcher->send_async_recv_decompressed(RPC_CMD_GET_TENSOR, comp_request, sizeof(*comp_request), data, size,
                                                  tensor->type);
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

static void add_tensor(ggml_tensor * tensor, const ggml_cgraph * cgraph, const rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor result = serialize_tensor(tensor, dispatcher);
    const size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    if (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        result.use_count = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(result);
}

static uint8_t * serialize_graph(uint32_t device, const ggml_cgraph * cgraph, const rpc_dispatcher * dispatcher, size_t * output_size) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_size = 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    uint8_t * output = new uint8_t[*output_size]();
    uint8_t * dest = output;
    memcpy(dest, &device, sizeof(device));
    dest += sizeof(device);
    memcpy(dest, &cgraph->uid, sizeof(cgraph->uid));
    dest += sizeof(cgraph->uid);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes * sizeof(uint64_t);
    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
    return output;
}

void rpc_dispatcher::graph_compute(uint32_t device, const ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    auto & device_graph_uids = graph_uids[device];
    bool reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    if (reuse) {
        auto request = std::make_shared<rpc_msg_graph_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_GRAPH_RECOMPUTE, request, sizeof(*request));
    } else {
        if (cgraph->uid != 0) {
            if (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_graph_uids.size());
                device_graph_uids.clear();
            }
            device_graph_uids.insert(cgraph->uid);
        }
        size_t input_size = 0;
        uint8_t * input = serialize_graph(device, cgraph, this, &input_size);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_size(input_size).c_str());
        std::shared_ptr<uint8_t> input_ptr(input, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_GRAPH_COMPUTE, input_ptr, input_size);
    }
}

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->graph_compute(rpc_ctx->device, cgraph);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_record(event);
}

static void ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_UNUSED(backend);
    GGML_UNUSED(event);
}

static void ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

static void ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_size(dispatcher, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_size(alignment).c_str(), rpc_log_format_size(max_size).c_str());
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    auto dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    return backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto dispatcher = get_dispatcher(endpoint);
    auto request = std::make_shared<rpc_msg_get_device_memory_req>();
    request->device = device;
    rpc_msg_get_device_memory_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_MEMORY, request, sizeof(*request), &response, sizeof(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    *free = response.free_mem;
    *total = response.total_mem;
}

// RPC server-side implementation

class rpc_server {
public:
    // `cache` is the server's disk cache for weights, nullptr when disabled
    rpc_server(std::vector<ggml_backend_t> all_backends, rpc_disk_cache * cache)
        : backends(std::move(all_backends)), cache(cache) {
        stored_graphs.resize(backends.size());
        comm_states.resize(backends.size());
    }

    // compression level negotiated with the client in the HELLO handshake
    uint8_t comp_level = RPC_COMP_NONE;
    ~rpc_server();

    // with compression negotiated, prefixes `response` (the plain tensor
    // bytes) with a flags byte and compresses it if that makes it smaller
    void compress_response(std::vector<uint8_t> & response, uint8_t level, const rpc_tensor & tensor) const;

    void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool memset_tensor(const rpc_msg_memset_tensor_req & request);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool set_tensor_2d(const std::vector<uint8_t> & input);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    void cache_list(std::vector<uint8_t> & response);
    bool set_tensor_cached(const rpc_msg_set_tensor_cached_req & request, rpc_msg_set_tensor_cached_rsp & response);
    // `level` is the most aggressive compression level the client allows for
    // the response (capped by the negotiated level)
    bool get_tensor(const rpc_msg_get_tensor_req & request, uint8_t level, std::vector<uint8_t> & response);
    bool get_tensor_2d(const rpc_msg_get_tensor_2d_req & request, uint8_t level, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response);
    bool comm_allreduce(const rpc_msg_comm_allreduce_req & request);
    bool comm_free(const rpc_msg_comm_free_req & request);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

private:
    void sync_all_backends();
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);


    // pairwise allreduce over a direct connection to the peer server
    struct comm_state {
        socket_ptr              peer;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_buffer_ptr scratch;
        size_t                  scratch_size = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
    };

    std::vector<ggml_backend_t> backends;
    rpc_disk_cache * cache;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<comm_state> comm_states;
};

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (void*)tensor->buffer, tensor->data);
    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    return true;
}

bool rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_size(request.size).c_str(), response.remote_ptr,
                rpc_log_format_size(response.remote_size).c_str());
        buffers.insert(buffer);
    } else {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_size(request.size).c_str());
    }
    return true;
}

bool rpc_server::get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_size(alignment).c_str());
    response.alignment = alignment;
    return true;
}

bool rpc_server::get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_size(max_size).c_str());
    response.max_size = max_size;
    return true;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    for (auto & sgs : stored_graphs) {
        for (auto & sg : sgs) {
            sg.second.graph = nullptr;
        }
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

bool rpc_server::memset_tensor(const rpc_msg_memset_tensor_req & request) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }

    const uint64_t tensor_size = ggml_nbytes(tensor);
    if (request.offset > tensor_size || request.size > tensor_size - request.offset) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_size);
        return false;
    }

    const uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
    const uint64_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);
    if (request.tensor.data < buffer_start) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        return false;
    }
    const uint64_t data_offset = request.tensor.data - buffer_start;
    if (data_offset > buffer_size ||
        request.offset > buffer_size - data_offset ||
        request.size > buffer_size - data_offset - request.offset) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        return false;
    }
    if (tensor->buffer->iface.memset_tensor == nullptr) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        return false;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void *) tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    // Validate tensor type before using it
    if (tensor->type >= GGML_TYPE_COUNT) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        return nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    if (ggml_blck_size((enum ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        return nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    if (result == nullptr) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        return nullptr;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    if (result->buffer && ggml_nelements(result) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        if (tensor->data + tensor_size < tensor->data ||
            tensor->data < buffer_start || tensor->data + tensor_size > buffer_start + buffer_size) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(result->type),
                      result->ne[0], result->ne[1], result->ne[2], result->ne[3],
                      tensor->data, tensor->data + tensor_size, buffer_start, buffer_start + buffer_size);
            return nullptr;
        }
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}


bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
    sync_all_backends();
    // serialization format: | rpc_tensor | flags (1 byte) | offset (8 bytes) | data (size bytes) |
    // bit 1 of the flags byte marks a compressed data section.
    uint8_t  flags;
    uint64_t offset;
    const size_t header_size = sizeof(rpc_tensor) + sizeof(flags) + sizeof(offset);
    if (input.size() < header_size) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    memcpy(&flags,  input.data() + sizeof(rpc_tensor), sizeof(flags));
    memcpy(&offset, input.data() + sizeof(rpc_tensor) + sizeof(flags), sizeof(offset));
    const bool compressed = (flags & RPC_SET_TENSOR_FLAG_COMPRESSED) != 0;
    const size_t wire_size = input.size() - header_size;

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    // the size of the write: for a compressed payload, the uncompressed size
    // from its header. It is only trusted after the bounds check below, and
    // nothing is allocated before that.
    const void * data = input.data() + header_size;
    size_t size = wire_size;
    if (compressed) {
        uint64_t orig_size = 0;
        if (comp_level == RPC_COMP_NONE || !rpc_compressed_orig_size(data, wire_size, orig_size) || orig_size > SIZE_MAX) {
            LOG_ERROR("[%s] invalid compressed payload\n", __func__);
            return false;
        }
        size = (size_t) orig_size;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str());

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            return false;
        }
    }

    // decompress: everything downstream (hash cache, the write) works on the
    // plain bytes
    std::vector<uint8_t> plain;
    if (compressed) {
        try {
            plain.resize(size);
        } catch (const std::bad_alloc &) {
            LOG_ERROR("[%s] failed to allocate %zu bytes\n", __func__, size);
            return false;
        }
        if (!rpc_decompress_payload(plain.data(), size, data, wire_size, (enum ggml_type) in_tensor->type)) {
            LOG_ERROR("[%s] failed to decompress payload for tensor %s\n", __func__, in_tensor->name);
            return false;
        }
        LOG_DBG3("[%s] tensor %s decompressed %s -> %s\n", __func__, in_tensor->name,
                 rpc_log_format_size(wire_size).c_str(), rpc_log_format_size(size).c_str());
        data = plain.data();
    }

    ggml_backend_tensor_set(tensor, data, offset, size);
    if (cache && (flags & RPC_SET_TENSOR_FLAG_CACHE)) {
        // the entry is named from the server's own digest of the data, never
        // from anything the client claims
        cache->store(data, size);
    }
    return true;
}

bool rpc_server::set_tensor_2d(const std::vector<uint8_t> & input) {
    sync_all_backends();
    // serialization format: | rpc_tensor | offset (8) | size (8) | n_copies (8) | stride (8) | data |
    // When compression was negotiated, a flags byte follows the header and
    // the data section may be compressed (see rpc-compress.h). A peer that
    // negotiated RPC_COMP_NONE (or an older version) sends no flags byte.
    const size_t header_size = sizeof(rpc_tensor) + 4*sizeof(uint64_t);
    if (input.size() < header_size) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t header[4];
    memcpy(header, input.data() + sizeof(rpc_tensor), sizeof(header));
    const uint64_t offset   = header[0];
    const uint64_t size     = header[1];
    const uint64_t n_copies = header[2];
    const uint64_t stride   = header[3];

    if (n_copies == 0 || size == 0 || size > UINT64_MAX / n_copies) {
        return false;
    }
    const uint64_t payload_size = size * n_copies;

    // when compression is negotiated the flags byte is always present
    const bool has_flags = comp_level != RPC_COMP_NONE;
    const size_t data_off = header_size + (has_flags ? 1 : 0);
    if (input.size() < data_off) {
        return false;
    }
    uint8_t flags = has_flags ? input[header_size] : 0;
    const bool compressed = (flags & RPC_SET_TENSOR_FLAG_COMPRESSED) != 0;
    const uint64_t wire_size = input.size() - data_off;

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, offset,
            rpc_log_format_size(size).c_str(), n_copies, stride);

    // sanitize tensor->data
    {
        if (stride != 0 && n_copies - 1 > (UINT64_MAX - size) / stride) {
            return false;
        }
        const uint64_t span = (n_copies - 1)*stride + size;
        const uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        const uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data < p0 || in_tensor->data > p1 || offset > p1 - in_tensor->data || span > p1 - in_tensor->data - offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, in_tensor->data, offset, span, p0, p1);
            return false;
        }
        if (offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - offset) {
            LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, offset, span, ggml_nbytes(tensor));
            return false;
        }
    }

    // decompress the data section first: everything downstream works on the
    // plain bytes. The uncompressed size must match size*n_copies exactly.
    std::vector<uint8_t> plain;
    const void * data;
    if (compressed) {
        // with overlapping copies (stride < size) the bounds check above does
        // not limit size*n_copies, so cap it by the destination buffer before
        // allocating: a legitimate write never carries more than that
        if (wire_size < RPC_COMP_HEADER_SIZE || payload_size > ggml_backend_buffer_get_size(tensor->buffer)) {
            LOG_ERROR("[%s] invalid compressed payload\n", __func__);
            return false;
        }
        try {
            plain.resize(payload_size);
        } catch (const std::bad_alloc &) {
            LOG_ERROR("[%s] failed to allocate %" PRIu64 " bytes\n", __func__, payload_size);
            return false;
        }
        if (!rpc_decompress_payload(plain.data(), plain.size(), input.data() + data_off, wire_size,
                                    (enum ggml_type) in_tensor->type)) {
            LOG_ERROR("[%s] failed to decompress payload for tensor %s\n", __func__, in_tensor->name);
            return false;
        }
        data = plain.data();
        LOG_DBG3("[%s] tensor %s decompressed %s -> %s\n", __func__, in_tensor->name,
                 rpc_log_format_size(wire_size).c_str(), rpc_log_format_size(payload_size).c_str());
    } else {
        if (wire_size != payload_size) {
            return false;
        }
        data = input.data() + data_off;
    }
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride, size);
    return true;
}

// The previous cache protocol keyed entries by a client-supplied 64-bit FNV
// hash, which a client could collide on purpose. Such queries always miss;
// clients that use them still work, just without the cache.
bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response) {
    GGML_UNUSED(request);
    response.result = 0;
    return true;
}

void rpc_server::cache_list(std::vector<uint8_t> & response) {
    response.clear();
    if (cache == nullptr) {
        return;
    }
    const std::vector<rpc_digest> entries = cache->list();
    response.resize(entries.size() * sizeof(rpc_digest::b));
    for (size_t i = 0; i < entries.size(); ++i) {
        memcpy(response.data() + i*sizeof(rpc_digest::b), entries[i].b.data(), sizeof(rpc_digest::b));
    }
    LOG_DBG("[%s] %zu entries\n", __func__, entries.size());
}

bool rpc_server::set_tensor_cached(const rpc_msg_set_tensor_cached_req & request, rpc_msg_set_tensor_cached_rsp & response) {
    sync_all_backends();
    response.result = 0;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || request.size > (p1 - request.tensor.data - request.offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                      __func__, request.tensor.data, request.offset, request.size, p0, p1);
            return false;
        }
    }

    rpc_digest digest;
    memcpy(digest.b.data(), request.digest, sizeof(digest.b));
    std::vector<uint8_t> data;
    if (cache == nullptr || !cache->load(digest, request.size, data)) {
        // the client only asks for entries listed in this connection, which
        // are never evicted while it lasts, so this is a disk problem
        LOG_ERROR("[%s] cache entry %s (%" PRIu64 " bytes) not available\n", __func__, digest.hex().c_str(), request.size);
        return true;
    }
    LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", size: %s, cache hit %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), request.offset, rpc_log_format_size(request.size).c_str(), digest.hex().c_str());
    ggml_backend_tensor_set(tensor, data.data(), request.offset, request.size);
    response.result = 1;
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        return false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (void*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        if (!buffer) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }

    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, uint8_t level, std::vector<uint8_t> & response) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, request.offset,
             rpc_log_format_size(request.size).c_str());

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                return false;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    compress_response(response, level, request.tensor);
    return true;
}

void rpc_server::compress_response(std::vector<uint8_t> & response, uint8_t level, const rpc_tensor & tensor) const {
    if (comp_level == RPC_COMP_NONE) {
        return;
    }
    std::vector<uint8_t> wire;
    const uint8_t codec = rpc_compress_payload(wire, response.data(), response.size(), (enum ggml_type) tensor.type,
                                               tensor.ne[0], std::min(level, comp_level));
    if (codec != RPC_CODEC_RAW) {
        LOG_DBG3("[%s] tensor %s compressed %s -> %s (codec %u)\n", __func__, tensor.name,
                 rpc_log_format_size(response.size()).c_str(), rpc_log_format_size(wire.size()).c_str(), codec);
        // the compressed payload starts with its own header; make room for the flags byte
        wire.insert(wire.begin(), RPC_SET_TENSOR_FLAG_COMPRESSED);
        response.swap(wire);
    } else {
        response.insert(response.begin(), 0);
    }
}

bool rpc_server::get_tensor_2d(const rpc_msg_get_tensor_2d_req & request, uint8_t level, std::vector<uint8_t> & response) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        if (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            return false;
        }
        if (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            return false;
        }
        const uint64_t span = (request.n_copies - 1)*request.stride + request.size;
        const uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        const uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data < p0 || request.tensor.data > p1 || request.offset > p1 - request.tensor.data ||
                span > p1 - request.tensor.data - request.offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.data, request.offset, span, p0, p1);
            return false;
        }
        if (request.offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - request.offset) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.offset, span, ggml_nbytes(tensor));
            return false;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.data(), request.offset, request.size, request.n_copies, request.stride, request.size);
    compress_response(response, level, request.tensor);
    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        return false;
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    if (it_ptr == tensor_ptrs.end()) {
        return nullptr;
    }
    const rpc_tensor * tensor = it_ptr->second;

    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    if (result->buffer == nullptr && result->data != nullptr) {
        LOG_ERROR("[%s] invalid data ptr\n", __func__);
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        if (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } else {
            result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            if (result->src[i] == nullptr) {
                LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                return nullptr;
            }
        }
    }

    // Handle view_src similarly
    if (tensor->view_src == 0) {
        result->view_src = nullptr;
    } else {
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        if (result->view_src == nullptr) {
            LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            return nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    return result;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t)) {
        return false;
    }
    const uint8_t * src = input.data();
    uint32_t device;
    memcpy(&device, src, sizeof(device));
    src += sizeof(device);
    if (device >= backends.size()) {
        return false;
    }
    uint64_t uid;
    memcpy(&uid, src, sizeof(uid));
    src += sizeof(uid);
    uint32_t n_nodes;
    memcpy(&n_nodes, src, sizeof(n_nodes));
    src += sizeof(n_nodes);
    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes*sizeof(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, sizeof(n_tensors));
    src += sizeof(n_tensors);
    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    if (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    if (sg.buffer.size() < buf_size) {
        sg.buffer.resize(buf_size);
    }
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ sg.buffer.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    const auto t_deserialize = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        if (graph->nodes[i] == nullptr && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            return false;
        }
        if (graph->nodes[i] != nullptr) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            const size_t hash_pos = ggml_hash_insert(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_count;
        }
    }
    const std::chrono::duration<double, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    return true;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    auto it = stored_graphs[device].find(request.uid);
    if (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        return false;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    return true;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
void rpc_server::sync_all_backends() {
    for (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

// The comm link between two servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
bool rpc_server::comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response) {
    response.ok = 0;
    if (request.device >= backends.size() || request.world != 2 || request.rank >= request.world) {
        return true;
    }
    comm_state & state = comm_states[request.device];
    if (state.peer != nullptr) {
        response.ok = 1;
        return true;
    }
    uint8_t local_caps[RPC_CONN_CAPS_SIZE] = {};
    uint8_t remote_caps[RPC_CONN_CAPS_SIZE] = {};
    if (request.rank == 0) {
        socket_ptr srv = socket_t::create_server("0.0.0.0", request.port);
        if (srv == nullptr) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.port);
            return true;
        }
        state.peer = srv->accept();
        if (state.peer == nullptr) {
            return true;
        }
        if (!state.peer->recv_data(remote_caps, sizeof(remote_caps))) {
            state.peer = nullptr;
            return true;
        }
        state.peer->get_caps(local_caps);
        if (!state.peer->send_data(local_caps, sizeof(local_caps))) {
            state.peer = nullptr;
            return true;
        }
        state.peer->update_caps(remote_caps);
    } else {
        const std::string host(request.host, strnlen(request.host, sizeof(request.host)));
        // rank 0 may not be listening yet, retry for a few seconds
        for (int i = 0; i < 100 && state.peer == nullptr; i++) {
            state.peer = socket_t::connect(host.c_str(), request.port);
            if (state.peer == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        if (state.peer == nullptr) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
            return true;
        }
        state.peer->get_caps(local_caps);
        if (!state.peer->send_data(local_caps, sizeof(local_caps)) ||
            !state.peer->recv_data(remote_caps, sizeof(remote_caps))) {
            state.peer = nullptr;
            return true;
        }
        state.peer->update_caps(remote_caps);
    }
    state.rank  = request.rank;
    state.world = request.world;
    LOG_INFO("[%s] device %u joined pairwise comm as rank %u\n", __func__, request.device, request.rank);
    response.ok = 1;
    return true;
}

bool rpc_server::comm_allreduce(const rpc_msg_comm_allreduce_req & request) {
    if (request.device >= backends.size()) {
        return false;
    }
    comm_state & state = comm_states[request.device];
    if (state.peer == nullptr) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        return false;
    }
    ggml_backend_t backend = backends[request.device];

    size_t ctx_size = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, false);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * t_dst = deserialize_tensor(ctx, &request.tensor);
    if (t_dst == nullptr || t_dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    const size_t  nbytes = ggml_nbytes(t_dst);
    const int64_t ne     = ggml_nelements(t_dst);
    if (nbytes == 0) {
        return true;
    }
    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    const bool   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    const size_t wire_bytes = wire_bf16 ? (size_t) ne*2 : nbytes;
    const size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    if (state.scratch_size < need) {
        state.scratch.reset(ggml_backend_alloc_buffer(backend, need));
        state.scratch_size = need;
    }
    char * scratch_base = (char *) ggml_backend_buffer_get_base(state.scratch.get());
    state.send_buf.resize(wire_bytes);
    state.recv_buf.resize(wire_bytes);

    auto new_scratch_tensor = [&](ggml_type type, size_t offset) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->buffer = state.scratch.get();
        t->data   = scratch_base + offset;
        return t;
    };
    auto new_cpy_node = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->buffer = dst->buffer;
        t->data   = dst->data;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        return t;
    };
    auto compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, false);
        graph->nodes[0] = n0;
        graph->nodes[1] = n1;
        graph->n_nodes  = n1 != nullptr ? 2 : 1;
        ggml_status status = ggml_backend_graph_compute_async(backend, graph);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    };

    // wait for the pending subgraph that produced this partial
    ggml_backend_synchronize(backend);

    ggml_tensor * t_wire_send = nullptr;
    ggml_tensor * t_wire_recv = nullptr;
    if (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        compute_nodes(new_cpy_node(t_dst, t_wire_send), nullptr);
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(t_wire_send, state.send_buf.data(), 0, wire_bytes);
    } else {
        ggml_backend_tensor_get(t_dst, state.send_buf.data(), 0, wire_bytes);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    if (state.rank == 0) {
        if (!state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush() ||
            !state.peer->recv_data(state.recv_buf.data(), wire_bytes)) {
            return false;
        }
    } else {
        if (!state.peer->recv_data(state.recv_buf.data(), wire_bytes) ||
            !state.peer->send_data(state.send_buf.data(), wire_bytes) || !state.peer->flush()) {
            return false;
        }
    }

    ggml_tensor * t_peer = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast = nullptr;
    if (wire_bf16) {
        ggml_backend_tensor_set(t_wire_recv, state.recv_buf.data(), 0, wire_bytes);
        t_cast = new_cpy_node(t_wire_recv, t_peer);
    } else {
        ggml_backend_tensor_set(t_peer, state.recv_buf.data(), 0, wire_bytes);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ADD;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->buffer = t_dst->buffer;
    t_red->data   = t_dst->data;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    if (t_cast != nullptr) {
        compute_nodes(t_cast, t_red);
    } else {
        compute_nodes(t_red, nullptr);
    }
    return true;
}

bool rpc_server::comm_free(const rpc_msg_comm_free_req & request) {
    if (request.device >= backends.size()) {
        return false;
    }
    comm_states[request.device] = comm_state();
    return true;
}

bool rpc_server::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    size_t free, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &free, &total);
    response.free_mem = free;
    response.total_mem = total;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    return true;
}

rpc_server::~rpc_server() {
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, rpc_disk_cache * cache,
                             socket_ptr sock) {
    rpc_server server(backends, cache);
    if (cache != nullptr) {
        cache->begin_session();
    }
    uint8_t cmd;
    if (!sock->recv_data(&cmd, 1)) {
        return;
    }
    if (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        return;
    }

    // Read input_size and validate protocol version
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return;
    }

    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_input_size, sizeof(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        return;
    }

    rpc_msg_hello_req req = {};
    if (!sock->recv_data(&req, sizeof(req))) {
        return;
    }

    rpc_msg_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    const uint8_t local_comp = rpc_compress_env_level();
    rpc_caps_set_comp(rsp.conn_caps, local_comp);
    rpc_caps_set_features(rsp.conn_caps, cache != nullptr ? RPC_FEAT_CACHE : 0);
    if (!send_msg(sock, &rsp, sizeof(rsp))) {
        return;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);

    // compression is negotiated down to the level both sides support
    server.comp_level = rpc_comp_negotiate(local_comp, rpc_caps_get_comp(req.conn_caps));
    LOG_INFO("client connected, transport: %s, compression: %s\n", sock->transport_name(),
             rpc_compress_level_name(server.comp_level));
    while (true) {
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((enum rpc_cmd)cmd));
        switch (cmd) {
            case RPC_CMD_HELLO: {
                // HELLO command is handled above
                return;
            }
            case RPC_CMD_DEVICE_COUNT: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_device_count_rsp response;
                response.device_count = backends.size();
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_alloc_buffer_rsp response;
                if (!server.alloc_buffer(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alloc_size_rsp response;
                if (!server.get_alloc_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                rpc_msg_get_alignment_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alignment_rsp response;
                if (!server.get_alignment(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                rpc_msg_get_max_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_max_size_rsp response;
                if (!server.get_max_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_buffer_get_base_rsp response;
                if (!server.buffer_get_base(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.free_buffer(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.buffer_clear(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_MEMSET_TENSOR: {
                rpc_msg_memset_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.memset_tensor(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_2D: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor_2d(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR_2D: {
                // with compression negotiated the request carries the level
                // the client allows for the response
                rpc_msg_get_tensor_2d_comp_req request = {};
                if (server.comp_level != RPC_COMP_NONE) {
                    if (!recv_msg(sock, &request, sizeof(request))) {
                        return;
                    }
                } else if (!recv_msg(sock, &request.req, sizeof(request.req))) {
                    return;
                }
                if (request.level >= RPC_COMP_COUNT) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor_2d(request.req, (uint8_t) request.level, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_CACHE_LIST: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input) || !input.empty()) {
                    return;
                }
                std::vector<uint8_t> response;
                server.cache_list(response);
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_CACHED: {
                rpc_msg_set_tensor_cached_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_cached_rsp response;
                if (!server.set_tensor_cached(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_HASH: {
                rpc_msg_set_tensor_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_hash(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sock, &request,sizeof(request))) {
                    return;
                }
                if (!server.init_tensor(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                // with compression negotiated the request carries the level
                // the client allows for the response
                rpc_msg_get_tensor_comp_req request = {};
                if (server.comp_level != RPC_COMP_NONE) {
                    if (!recv_msg(sock, &request, sizeof(request))) {
                        return;
                    }
                } else if (!recv_msg(sock, &request.req, sizeof(request.req))) {
                    return;
                }
                if (request.level >= RPC_COMP_COUNT) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor(request.req, (uint8_t) request.level, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.graph_compute(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_msg_graph_recompute_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.graph_recompute(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_COMM_INIT: {
                rpc_msg_comm_init_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_comm_init_rsp response;
                if (!server.comm_init(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_COMM_ALLREDUCE: {
                rpc_msg_comm_allreduce_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.comm_allreduce(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_COMM_FREE: {
                rpc_msg_comm_free_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.comm_free(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_msg_get_device_memory_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_device_memory_rsp response;
                if (!server.get_device_memory(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            default: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((enum rpc_cmd)cmd));
                return;
            }
        }
    }
}

void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    if (n_devices == 0 || devices == nullptr) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    // size limit in GiB from GGML_RPC_CACHE_SIZE (set by rpc-server --cache-size); 0 = no fixed limit
    std::unique_ptr<rpc_disk_cache> cache;
    if (cache_dir != nullptr) {
        const char * env = std::getenv("GGML_RPC_CACHE_SIZE");
        const double gib = env ? std::atof(env) : 0.0;
        cache = std::make_unique<rpc_disk_cache>(cache_dir, gib > 0 ? (uint64_t) (gib * 1073741824.0) : 0);
    }
    LOG_INFO("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            return;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }

#ifdef GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    LOG_INFO("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    if (server_socket == nullptr) {
        LOG_ERROR("Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            LOG_ERROR("Failed to accept client connection\n");
            return;
        }
        LOG_INFO("Accepted client connection\n");
        rpc_serve_client(backends, cache.get(), client_socket);
        LOG_INFO("Client connection closed\n");
    }
    rpc_transport_shutdown();
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    return true;
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    return buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

static ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    return dispatcher->event_new(dev);
}

static void ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

static void ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

static const struct ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

struct ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->name.c_str() : "RPC";
}

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->devices.size() : 0;
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    if (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } else {
        GGML_ASSERT(index < ctx->devices.size());
        return ctx->devices[index];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
struct ggml_backend_rpc_comm_context {
    struct rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

static void ggml_backend_rpc_comm_free(void * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    if (comm_ctx == nullptr) {
        return;
    }
    for (const auto & rank : comm_ctx->ranks) {
        auto request = std::make_shared<rpc_msg_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, sizeof(*request));
        rank.dispatcher->busy_spin_release();
    }
    delete comm_ctx;
}

static void * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    if (n_backends != 2 || std::getenv("GGML_RPC_NO_COMM") != nullptr) {
        return nullptr;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_rpc(backends[i])) {
            return nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        for (const auto & rank : ranks) {
            if (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                return nullptr;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    int port0;
    if (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        return nullptr;
    }
    const uint32_t comm_port = (uint32_t) port0 + 1000;
    if (host0.size() >= 64) {
        return nullptr;
    }

    for (const auto & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_msg_comm_init_rsp> responses(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        auto request = std::make_shared<rpc_msg_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->port   = comm_port;
        if (i > 0) {
            memcpy(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, sizeof(*request), &responses[i], sizeof(responses[i]));
    }
    for (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    bool ok = true;
    for (size_t i = 0; i < n_backends; i++) {
        if (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = false;
        }
    }
    if (!ok) {
        for (const auto & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        return nullptr;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    return new ggml_backend_rpc_comm_context{std::move(ranks)};
}

static bool ggml_backend_rpc_comm_allreduce_tensor(void * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    if (comm_ctx == nullptr) {
        return false;
    }
    const size_t n_ranks = comm_ctx->ranks.size();
    const int64_t ne = ggml_nelements(tensors[0]);
    if (ne == 0) {
        return true;
    }
    for (size_t i = 0; i < n_ranks; i++) {
        if (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensors[i]->buffer)) {
            return false;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
    }
    for (size_t i = 0; i < n_ranks; i++) {
        auto request = std::make_shared<rpc_msg_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, sizeof(*request));
    }
    return true;
}

// Model loaders that store weight digests (see llama-model-loader.cpp) use
// these two, looked up through the registry, to skip reading cached weights.

// Sets a weight tensor from the server's cache entry with this digest (32
// bytes), without its data. Returns false when the tensor is not in an RPC
// buffer, the server has no cache, or it has no such entry: the caller then
// uploads the data as usual.
static bool ggml_backend_rpc_set_tensor_cached(struct ggml_tensor * tensor, size_t offset, size_t size, const uint8_t * digest) {
    if (tensor->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensor->buffer)) {
        return false;
    }
    auto * ctx = (ggml_backend_rpc_buffer_context *) tensor->buffer->context;
    if (!rpc_tensor_cacheable(ctx->dispatcher, tensor, size)) {
        return false;
    }
    rpc_digest d;
    memcpy(d.b.data(), digest, sizeof(d.b));
    if (!ctx->dispatcher->cache_contains(d)) {
        return false;
    }
    rpc_send_set_tensor_cached(ctx->dispatcher, serialize_tensor(tensor), offset, size, d);
    LOG_DBG2("[%s] tensor: %s, cache hit %s (stored digest)\n", __func__, rpc_log_format_tensor(tensor).c_str(), d.hex().c_str());
    return true;
}

// Uploads like ggml_backend_tensor_set and writes the cache digest it computed
// to `digest` (32 bytes). Returns false when no digest was computed (not an
// RPC buffer, no server cache, or a tensor too small to cache); the tensor is
// uploaded either way.
static bool ggml_backend_rpc_set_tensor_digest(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, uint8_t * digest) {
    if (tensor->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_tensor_set(tensor, data, offset, size);
        return false;
    }
    auto * ctx = (ggml_backend_rpc_buffer_context *) tensor->buffer->context;
    const bool cacheable = rpc_tensor_cacheable(ctx->dispatcher, tensor, size);
    rpc_digest d;
    rpc_buffer_set_tensor_impl(tensor->buffer, tensor, data, offset, size, cacheable ? &d : nullptr);
    if (cacheable) {
        memcpy(digest, d.b.data(), sizeof(d.b));
    }
    return cacheable;
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        return (void *)ggml_backend_rpc_add_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        return (void *)ggml_backend_rpc_start_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_set_tensor_cached") == 0) {
        return (void *)ggml_backend_rpc_set_tensor_cached;
    }
    if (std::strcmp(name, "ggml_backend_rpc_set_tensor_digest") == 0) {
        return (void *)ggml_backend_rpc_set_tensor_digest;
    }
    if (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_rpc_comm_init;
    }
    if (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_rpc_comm_free;
    }
    if (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static struct ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_rpc_reg;
}

static uint32_t ggml_backend_rpc_get_device_count(const char * endpoint) {
    auto dispatcher = get_dispatcher(endpoint);
    rpc_msg_device_count_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_COUNT, nullptr, 0, &response, sizeof(response));
    return response.device_count;
}

static const ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint) {
    static std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    static std::mutex mutex;
    static uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        return reg_map[endpoint];
    }
    uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_count);
    if (dev_count == 0) {
        return nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = new ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = new ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = new ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    return reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
