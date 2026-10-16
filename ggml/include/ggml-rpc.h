#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    8
#define RPC_PROTO_MINOR_VERSION    1
#define RPC_PROTO_PATCH_VERSION    0

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 101, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

// weight cache helpers, obtained with ggml_backend_reg_get_proc_address(ggml_backend_rpc_reg(), name):
// "ggml_backend_rpc_set_tensor_cached": sets a weight from the server's cache entry with this 32-byte digest,
//     without its data; false when not possible (the caller then uploads the data)
// "ggml_backend_rpc_set_tensor_digest": uploads like ggml_backend_tensor_set and writes the 32-byte cache
//     digest it computed; false when none was computed. from_cache (optional) tells whether the server set
//     the tensor from its cache
typedef bool (*ggml_backend_rpc_set_tensor_cached_t)(struct ggml_tensor * tensor, size_t offset, size_t size, const uint8_t * digest);
typedef bool (*ggml_backend_rpc_set_tensor_digest_t)(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size, uint8_t * digest, bool * from_cache);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus
}
#endif
