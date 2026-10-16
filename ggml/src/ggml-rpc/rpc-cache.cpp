// Server-side weight cache for the RPC backend. See rpc-cache.h for the design.

#include "rpc-cache.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__)) && !defined(GGML_RPC_NO_SHANI)
#define RPC_HAVE_SHANI 1
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace fs = std::filesystem;

// ------------------------------------------------------------ SHA-256 ----

namespace {

struct sha256_ctx {
    uint32_t state[8];
    uint64_t n_bytes;
    uint8_t  buf[64];
    size_t   n_buf;
};

static const uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void sha256_init(sha256_ctx & c) {
    static const uint32_t init[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };
    memcpy(c.state, init, sizeof(init));
    c.n_bytes = 0;
    c.n_buf   = 0;
}

static void sha256_block(uint32_t state[8], const uint8_t * p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (uint32_t) p[4*i] << 24 | (uint32_t) p[4*i + 1] << 16 | (uint32_t) p[4*i + 2] << 8 | p[4*i + 3];
    }
    for (int i = 16; i < 64; ++i) {
        const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
        const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = h + s1 + ch + SHA256_K[i] + w[i];
        const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = s0 + mj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

static void sha256_blocks_portable(uint32_t state[8], const uint8_t * p, size_t n_blocks) {
    for (size_t i = 0; i < n_blocks; ++i) {
        sha256_block(state, p + 64*i);
    }
}

#ifdef RPC_HAVE_SHANI
// SHA-256 with the x86 SHA extensions (SHA-NI): several times faster than the
// portable code. Rounds go four at a time; w[] holds the last 16 message words.
__attribute__((target("sha,sse4.1")))
static void sha256_blocks_shani(uint32_t state[8], const uint8_t * p, size_t n_blocks) {
    const __m128i bswap = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);

    __m128i tmp    = _mm_loadu_si128((const __m128i *) &state[0]);
    __m128i state1 = _mm_loadu_si128((const __m128i *) &state[4]);
    tmp    = _mm_shuffle_epi32(tmp, 0xB1);          // CDAB
    state1 = _mm_shuffle_epi32(state1, 0x1B);       // EFGH
    __m128i state0 = _mm_alignr_epi8(tmp, state1, 8);   // ABEF
    state1 = _mm_blend_epi16(state1, tmp, 0xF0);    // CDGH

    for (size_t blk = 0; blk < n_blocks; ++blk, p += 64) {
        const __m128i abef = state0;
        const __m128i cdgh = state1;
        __m128i w[4];
        for (int g = 0; g < 16; ++g) {
            if (g < 4) {
                w[g] = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) (p + 16*g)), bswap);
            }
            const __m128i cur = w[g & 3];
            __m128i msg = _mm_add_epi32(cur, _mm_loadu_si128((const __m128i *) &SHA256_K[4*g]));
            state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
            if (g >= 3 && g <= 14) {
                const __m128i t = _mm_alignr_epi8(cur, w[(g - 1) & 3], 4);
                w[(g + 1) & 3] = _mm_sha256msg2_epu32(_mm_add_epi32(w[(g + 1) & 3], t), cur);
            }
            msg    = _mm_shuffle_epi32(msg, 0x0E);
            state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
            if (g >= 1 && g <= 12) {
                w[(g - 1) & 3] = _mm_sha256msg1_epu32(w[(g - 1) & 3], cur);
            }
        }
        state0 = _mm_add_epi32(state0, abef);
        state1 = _mm_add_epi32(state1, cdgh);
    }

    tmp    = _mm_shuffle_epi32(state0, 0x1B);       // FEBA
    state1 = _mm_shuffle_epi32(state1, 0xB1);       // DCHG
    state0 = _mm_blend_epi16(tmp, state1, 0xF0);    // DCBA
    state1 = _mm_alignr_epi8(state1, tmp, 8);       // ABEF
    _mm_storeu_si128((__m128i *) &state[0], state0);
    _mm_storeu_si128((__m128i *) &state[4], state1);
}

static bool cpu_has_shani() {
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
        return false;
    }
    const bool sha = (b >> 29) & 1;
    if (!__get_cpuid(1, &a, &b, &c, &d)) {
        return false;
    }
    const bool sse41 = (c >> 19) & 1;
    return sha && sse41;
}
#endif

using sha256_blocks_fn = void (*)(uint32_t *, const uint8_t *, size_t);

static sha256_blocks_fn sha256_blocks_select() {
#ifdef RPC_HAVE_SHANI
    if (cpu_has_shani()) {
        return sha256_blocks_shani;
    }
#endif
    return sha256_blocks_portable;
}

static const sha256_blocks_fn sha256_blocks = sha256_blocks_select();

static void sha256_update(sha256_ctx & c, const void * data, size_t size) {
    const uint8_t * p = (const uint8_t *) data;
    c.n_bytes += size;
    if (c.n_buf > 0) {
        const size_t n = std::min(size, 64 - c.n_buf);
        memcpy(c.buf + c.n_buf, p, n);
        c.n_buf += n;
        p += n;
        size -= n;
        if (c.n_buf == 64) {
            sha256_blocks(c.state, c.buf, 1);
            c.n_buf = 0;
        }
    }
    if (size >= 64) {
        const size_t n_blocks = size / 64;
        sha256_blocks(c.state, p, n_blocks);
        p    += 64*n_blocks;
        size -= 64*n_blocks;
    }
    if (size > 0) {
        memcpy(c.buf, p, size);
        c.n_buf = size;
    }
}

static void sha256_final(sha256_ctx & c, uint8_t out[32]) {
    const uint64_t bits = c.n_bytes * 8;
    const uint8_t pad = 0x80;
    sha256_update(c, &pad, 1);
    const uint8_t zero = 0;
    while (c.n_buf != 56) {
        sha256_update(c, &zero, 1);
    }
    uint8_t len[8];
    for (int i = 0; i < 8; ++i) {
        len[i] = (uint8_t) (bits >> (56 - 8*i));
    }
    sha256_update(c, len, 8);
    for (int i = 0; i < 8; ++i) {
        out[4*i]     = (uint8_t) (c.state[i] >> 24);
        out[4*i + 1] = (uint8_t) (c.state[i] >> 16);
        out[4*i + 2] = (uint8_t) (c.state[i] >> 8);
        out[4*i + 3] = (uint8_t) (c.state[i]);
    }
}

} // namespace

void rpc_sha256(const void * data, size_t size, uint8_t out[32]) {
    sha256_ctx c;
    sha256_init(c);
    sha256_update(c, data, size);
    sha256_final(c, out);
}

// ------------------------------------------------------------- digest ----

static constexpr size_t RPC_CACHE_CHUNK = 4u << 20;
static const char       RPC_CACHE_TAG[] = "ggml-rpc-cache-v1";

rpc_digest rpc_cache_digest(const void * data, size_t size) {
    const size_t n_chunks = std::max<size_t>(1, (size + RPC_CACHE_CHUNK - 1) / RPC_CACHE_CHUNK);
    std::vector<uint8_t> leaves(n_chunks * 32);

    auto hash_chunks = [&](size_t c0, size_t c1) {
        for (size_t c = c0; c < c1; ++c) {
            const size_t off = c * RPC_CACHE_CHUNK;
            const size_t len = std::min(RPC_CACHE_CHUNK, size - std::min(size, off));
            rpc_sha256((const uint8_t *) data + off, len, leaves.data() + 32*c);
        }
    };
    const size_t n_threads = std::min<size_t>({ n_chunks, 8, std::max(1u, std::thread::hardware_concurrency() / 2) });
    if (n_threads <= 1) {
        hash_chunks(0, n_chunks);
    } else {
        std::vector<std::thread> threads;
        const size_t per = (n_chunks + n_threads - 1) / n_threads;
        for (size_t t = 1; t < n_threads; ++t) {
            const size_t c0 = std::min(n_chunks, t*per);
            const size_t c1 = std::min(n_chunks, c0 + per);
            threads.emplace_back(hash_chunks, c0, c1);
        }
        hash_chunks(0, std::min(n_chunks, per));
        for (auto & th : threads) {
            th.join();
        }
    }

    sha256_ctx c;
    sha256_init(c);
    sha256_update(c, RPC_CACHE_TAG, sizeof(RPC_CACHE_TAG));
    uint8_t len[8];
    for (int i = 0; i < 8; ++i) {
        len[i] = (uint8_t) ((uint64_t) size >> (8*i));
    }
    sha256_update(c, len, 8);
    sha256_update(c, leaves.data(), leaves.size());
    rpc_digest d;
    sha256_final(c, d.b.data());
    return d;
}

std::string rpc_digest::hex() const {
    static const char * digits = "0123456789abcdef";
    std::string s(64, '0');
    for (size_t i = 0; i < 32; ++i) {
        s[2*i]     = digits[b[i] >> 4];
        s[2*i + 1] = digits[b[i] & 0xf];
    }
    return s;
}

bool rpc_digest::from_hex(const std::string & s, rpc_digest & out) {
    if (s.size() != 64) {
        return false;
    }
    auto val = [](char ch) -> int {
        if (ch >= '0' && ch <= '9') return ch - '0';
        if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
        return -1;
    };
    for (size_t i = 0; i < 32; ++i) {
        const int hi = val(s[2*i]);
        const int lo = val(s[2*i + 1]);
        if (hi < 0 || lo < 0) {
            return false;
        }
        out.b[i] = (uint8_t) (hi << 4 | lo);
    }
    return true;
}

// ---------------------------------------------------------- disk cache ----

// last-use time in file clock ticks: full resolution, so entries used within
// the same second keep their order
static int64_t file_time_ticks(fs::file_time_type t) {
    return (int64_t) t.time_since_epoch().count();
}

static bool is_hex(const std::string & s) {
    return std::all_of(s.begin(), s.end(), [](char ch) { return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'); });
}

rpc_disk_cache::rpc_disk_cache(const std::string & dir, uint64_t max_bytes) : dir(dir), max_bytes(max_bytes) {
    std::error_code ec;
    size_t n_removed = 0;
    for (const auto & it : fs::directory_iterator(dir, ec)) {
        if (!it.is_regular_file(ec)) {
            continue;
        }
        const std::string name = it.path().filename().string();
        rpc_digest d;
        if (rpc_digest::from_hex(name, d)) {
            const uint64_t size = it.file_size(ec);
            index[d] = { size, file_time_ticks(it.last_write_time(ec)) };
            total += size;
        } else if ((name.size() == 16 && is_hex(name)) || (name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0)) {
            // entries of the previous (64-bit FNV) cache format can never be
            // used again, and .tmp files are leftovers of interrupted writes
            fs::remove(it.path(), ec);
            n_removed++;
        }
    }
    LOG_INFO("  cache entries  : %zu (%.2f GiB)\n", index.size(), total / 1073741824.0);
    if (max_bytes > 0) {
        LOG_INFO("  cache limit    : %.2f GiB, keeping 2 GiB of disk free\n", max_bytes / 1073741824.0);
    } else {
        LOG_INFO("  cache limit    : none, keeping 2 GiB of disk free\n");
    }
    if (n_removed > 0) {
        LOG_INFO("  cache cleanup  : removed %zu stale files\n", n_removed);
    }
}

std::string rpc_disk_cache::path_of(const rpc_digest & digest) const {
    return (fs::path(dir) / digest.hex()).string();
}

void rpc_disk_cache::begin_session() {
    std::lock_guard<std::mutex> lock(mutex);
    protected_entries.clear();
    warned_full = false;
}

std::vector<rpc_digest> rpc_disk_cache::list() {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<rpc_digest> out;
    out.reserve(index.size());
    for (const auto & [d, e] : index) {
        out.push_back(d);
        protected_entries.insert(d);
    }
    return out;
}

void rpc_disk_cache::touch(const rpc_digest & digest) {
    std::error_code ec;
    const auto now = fs::file_time_type::clock::now();
    fs::last_write_time(path_of(digest), now, ec);
    auto it = index.find(digest);
    if (it != index.end()) {
        it->second.last_use = file_time_ticks(now);
    }
}

bool rpc_disk_cache::load(const rpc_digest & digest, uint64_t size, std::vector<uint8_t> & out) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = index.find(digest);
    if (it == index.end() || it->second.size != size) {
        return false;
    }
    std::ifstream f(path_of(digest), std::ios::binary);
    if (!f) {
        // the file disappeared behind our back
        total -= it->second.size;
        index.erase(it);
        return false;
    }
    try {
        out.resize(size);
    } catch (const std::bad_alloc &) {
        return false;
    }
    f.read((char *) out.data(), (std::streamsize) size);
    if ((uint64_t) f.gcount() != size) {
        return false;
    }
    touch(digest);
    protected_entries.insert(digest);
    return true;
}

// free disk space the cache always leaves
static constexpr uint64_t RPC_CACHE_DISK_RESERVE = 2ull << 30;

bool rpc_disk_cache::ensure_space(uint64_t need) {
    while (true) {
        std::error_code ec;
        const fs::space_info space = fs::space(dir, ec);
        const bool over_limit = max_bytes > 0 && total + need > max_bytes;
        const bool low_disk   = !ec && space.available < need + RPC_CACHE_DISK_RESERVE;
        if (!over_limit && !low_disk) {
            return true;
        }
        // evict the least recently used entry this connection does not need
        auto victim = index.end();
        for (auto it = index.begin(); it != index.end(); ++it) {
            if (protected_entries.count(it->first) == 0 && (victim == index.end() || it->second.last_use < victim->second.last_use)) {
                victim = it;
            }
        }
        if (victim == index.end()) {
            return false;
        }
        LOG_DBG("[%s] evicting %s (%.1f MiB)\n", __func__, victim->first.hex().c_str(), victim->second.size / 1048576.0);
        fs::remove(path_of(victim->first), ec);
        total -= victim->second.size;
        index.erase(victim);
    }
}

bool rpc_disk_cache::store(const void * data, size_t size) {
    const rpc_digest digest = rpc_cache_digest(data, size);

    std::lock_guard<std::mutex> lock(mutex);
    if (index.count(digest)) {
        touch(digest);
        protected_entries.insert(digest);
        return true;
    }
    if (!ensure_space(size)) {
        if (!warned_full) {
            std::error_code ec;
            const fs::space_info space = fs::space(dir, ec);
            if (max_bytes > 0 && total + size > max_bytes) {
                LOG_WARN("[%s] cache size limit reached (%.2f of %.2f GiB in use by entries this client needs), "
                         "not caching more tensors for this client\n", __func__, total / 1073741824.0, max_bytes / 1073741824.0);
            } else {
                LOG_WARN("[%s] not enough free disk space for the cache (%.2f GiB free, keeping %.2f GiB free; cache uses %.2f GiB), "
                         "not caching more tensors for this client\n", __func__, space.available / 1073741824.0,
                         RPC_CACHE_DISK_RESERVE / 1073741824.0, total / 1073741824.0);
            }
            warned_full = true;
        }
        return false;
    }
    // write to a temporary file and rename it, so a crash never leaves a
    // truncated entry under a valid name
    const std::string path = path_of(digest);
    const std::string tmp  = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.write((const char *) data, (std::streamsize) size);
        if (!f.good()) {
            f.close();
            std::error_code ec;
            fs::remove(tmp, ec);
            LOG_ERROR("[%s] failed to write %s\n", __func__, tmp.c_str());
            return false;
        }
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        LOG_ERROR("[%s] failed to rename %s\n", __func__, tmp.c_str());
        return false;
    }
    index[digest] = { size, file_time_ticks(fs::file_time_type::clock::now()) };
    total += size;
    protected_entries.insert(digest);
    LOG_DBG("[%s] cached %s (%.1f MiB)\n", __func__, digest.hex().c_str(), size / 1048576.0);
    return true;
}
