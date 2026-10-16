// Compression for the RPC transport. See rpc-compress.h for the design.

#include "rpc-compress.h"
#include "log.h"

#include "ggml-impl.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <thread>

#ifdef GGML_RPC_HAVE_ZSTD
#include <zstd.h>
#endif

const char * rpc_compress_level_name(uint8_t level) {
    switch (level) {
        case RPC_COMP_NONE: return "none";
        case RPC_COMP_ZSTD: return "zstd";
        case RPC_COMP_Q8:   return "q8";
        case RPC_COMP_Q6:   return "q6";
        case RPC_COMP_Q4:   return "q4";
        default:            return "unknown";
    }
}

bool rpc_compress_level_parse(const char * str, uint8_t & level) {
    static const struct { const char * name; uint8_t level; } names[] = {
        { "none", RPC_COMP_NONE }, { "zstd", RPC_COMP_ZSTD },
        { "q8",   RPC_COMP_Q8   }, { "q6",   RPC_COMP_Q6   }, { "q4", RPC_COMP_Q4 },
        { "fp8",  RPC_COMP_Q8   }, { "fp4",  RPC_COMP_Q4   },
    };
    for (const auto & n : names) {
        if (strcmp(str, n.name) == 0) {
            level = n.level;
            return true;
        }
    }
    if (str[0] >= '0' && str[0] <= '9' && str[1] == '\0' && rpc_compress_level_valid(str[0] - '0')) {
        level = str[0] - '0';
        return true;
    }
    return false;
}

uint8_t rpc_compress_env_level() {
    static const uint8_t level = [] {
        const char * env = std::getenv("GGML_RPC_COMPRESSION");
        uint8_t lvl = RPC_COMP_ZSTD;   // lossless default, always safe
        if (env != nullptr && env[0] != '\0' && !rpc_compress_level_parse(env, lvl)) {
            LOG_WARN("invalid GGML_RPC_COMPRESSION=%s (none, zstd, q8, q6, q4), using zstd\n", env);
            lvl = RPC_COMP_ZSTD;
        }
#ifndef GGML_RPC_HAVE_ZSTD
        // the lossless fallback of every level needs zstd on both sides
        if (lvl != RPC_COMP_NONE) {
            LOG_WARN("RPC compression requested but ggml-rpc was built without zstd, disabling it\n");
        }
        lvl = RPC_COMP_NONE;
#endif
        return lvl;
    }();
    return level;
}

// ---------------------------------------------------------------- zstd ----

// zstd level for float payloads (after the byte shuffle) and for everything
// else (mostly quantized weights, sent once at load). Level 1 compresses
// shuffled activations as well as level 3 at twice the speed.
static constexpr int ZSTD_LEVEL_FLOAT = 1;
static constexpr int ZSTD_LEVEL_OTHER = 3;
// payloads at least this big are compressed by several zstd workers
static constexpr size_t ZSTD_MT_MIN_SIZE = 4u << 20;

#ifdef GGML_RPC_HAVE_ZSTD
struct zstd_cctx_deleter { void operator()(ZSTD_CCtx * c) const { ZSTD_freeCCtx(c); } };
struct zstd_dctx_deleter { void operator()(ZSTD_DCtx * d) const { ZSTD_freeDCtx(d); } };

// contexts are reused: creating one per payload costs more than compressing
// a small activation
static ZSTD_CCtx * zstd_cctx() {
    thread_local std::unique_ptr<ZSTD_CCtx, zstd_cctx_deleter> cctx(ZSTD_createCCtx());
    return cctx.get();
}

static ZSTD_DCtx * zstd_dctx() {
    thread_local std::unique_ptr<ZSTD_DCtx, zstd_dctx_deleter> dctx(ZSTD_createDCtx());
    return dctx.get();
}
#endif

// appends the zstd frame of src to dst
static bool zstd_compress_append(std::vector<uint8_t> & dst, const void * src, size_t src_size, int level) {
#ifdef GGML_RPC_HAVE_ZSTD
    ZSTD_CCtx * cctx = zstd_cctx();
    if (cctx == nullptr) {
        return false;
    }
    ZSTD_CCtx_reset(cctx, ZSTD_reset_session_and_parameters);
    ZSTD_CCtx_setParameter(cctx, ZSTD_c_compressionLevel, level);
    if (src_size >= ZSTD_MT_MIN_SIZE) {
        // fails harmlessly when libzstd was built without multithreading
        const int n_workers = (int) std::min(8u, std::max(1u, std::thread::hardware_concurrency() / 2));
        ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, n_workers);
    }
    const size_t off   = dst.size();
    const size_t bound = ZSTD_compressBound(src_size);
    dst.resize(off + bound);
    const size_t written = ZSTD_compress2(cctx, dst.data() + off, bound, src, src_size);
    if (ZSTD_isError(written)) {
        LOG_ERROR("[%s] zstd compress failed: %s\n", __func__, ZSTD_getErrorName(written));
        dst.resize(off);
        return false;
    }
    dst.resize(off + written);
    return true;
#else
    GGML_UNUSED(dst); GGML_UNUSED(src); GGML_UNUSED(src_size); GGML_UNUSED(level);
    return false;
#endif
}

static bool zstd_decompress(void * dst, size_t dst_size, const void * src, size_t src_size) {
#ifdef GGML_RPC_HAVE_ZSTD
    ZSTD_DCtx * dctx = zstd_dctx();
    if (dctx == nullptr) {
        return false;
    }
    // the output is bounded by dst_size: a frame that expands further fails
    const size_t rc = ZSTD_decompressDCtx(dctx, dst, dst_size, src, src_size);
    if (ZSTD_isError(rc)) {
        LOG_ERROR("[%s] zstd decompress failed: %s\n", __func__, ZSTD_getErrorName(rc));
        return false;
    }
    return rc == dst_size;
#else
    GGML_UNUSED(dst); GGML_UNUSED(dst_size); GGML_UNUSED(src); GGML_UNUSED(src_size);
    return false;
#endif
}

// decompresses a frame whose content size is not known by the caller, into
// a buffer of at most max_size bytes
static bool zstd_decompress_bounded(std::vector<uint8_t> & dst, const void * src, size_t src_size, size_t max_size) {
#ifdef GGML_RPC_HAVE_ZSTD
    const unsigned long long n = ZSTD_getFrameContentSize(src, src_size);
    if (n == ZSTD_CONTENTSIZE_UNKNOWN || n == ZSTD_CONTENTSIZE_ERROR || n > max_size) {
        return false;
    }
    dst.resize((size_t) n);
    return zstd_decompress(dst.data(), dst.size(), src, src_size);
#else
    GGML_UNUSED(dst); GGML_UNUSED(src); GGML_UNUSED(src_size); GGML_UNUSED(max_size);
    return false;
#endif
}

// byte shuffle: element i, byte b -> dst[b*n + i]
static void shuffle_bytes(uint8_t * dst, const uint8_t * src, size_t n, size_t es) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t b = 0; b < es; ++b) {
            dst[b*n + i] = src[i*es + b];
        }
    }
}

static void unshuffle_bytes(uint8_t * dst, const uint8_t * src, size_t n, size_t es) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t b = 0; b < es; ++b) {
            dst[i*es + b] = src[b*n + i];
        }
    }
}

// ------------------------------------------------------- float payloads ----

static size_t float_elem_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_BF16: return 2;
        default:             return 0;
    }
}

static void to_f32(const void * src, float * dst, size_t n, enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            memcpy(dst, src, n*sizeof(float));
            break;
        case GGML_TYPE_F16:
            for (size_t i = 0; i < n; ++i) {
                dst[i] = GGML_FP16_TO_FP32(((const ggml_fp16_t *) src)[i]);
            }
            break;
        case GGML_TYPE_BF16:
            for (size_t i = 0; i < n; ++i) {
                dst[i] = GGML_BF16_TO_FP32(((const ggml_bf16_t *) src)[i]);
            }
            break;
        default:
            GGML_ABORT("not a float type");
    }
}

static void from_f32(const float * src, void * dst, size_t n, enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            memcpy(dst, src, n*sizeof(float));
            break;
        case GGML_TYPE_F16:
            for (size_t i = 0; i < n; ++i) {
                ((ggml_fp16_t *) dst)[i] = GGML_FP32_TO_FP16(src[i]);
            }
            break;
        case GGML_TYPE_BF16:
            for (size_t i = 0; i < n; ++i) {
                ((ggml_bf16_t *) dst)[i] = GGML_FP32_TO_BF16(src[i]);
            }
            break;
        default:
            GGML_ABORT("not a float type");
    }
}

// runs fn(first_row, last_row) over n_rows, split across threads when the
// payload is big enough to amortize starting them
template <typename F>
static void parallel_rows(size_t n_rows, size_t row_len, F && fn) {
    static constexpr size_t MIN_ELEMS_PER_THREAD = 128*1024;
    const size_t max_threads = std::min<size_t>(8, std::max(1u, std::thread::hardware_concurrency() / 2));
    const size_t n_threads   = std::min({ max_threads, n_rows, std::max<size_t>(1, n_rows*row_len / MIN_ELEMS_PER_THREAD) });
    if (n_threads <= 1) {
        fn(0, n_rows);
        return;
    }
    std::vector<std::thread> threads;
    const size_t per = (n_rows + n_threads - 1) / n_threads;
    for (size_t t = 1; t < n_threads; ++t) {
        const size_t r0 = std::min(n_rows, t*per);
        const size_t r1 = std::min(n_rows, r0 + per);
        if (r0 < r1) {
            threads.emplace_back([&fn, r0, r1] { fn(r0, r1); });
        }
    }
    fn(0, std::min(n_rows, per));
    for (auto & th : threads) {
        th.join();
    }
}

// ------------------------------------------------------------- e4m3 ----
//
// e4m3 (OCP FP8): sign bit 7, exponent bits 6-3 (bias 7), mantissa bits 2-0,
// no infinities, 0x7f/0xff are NaN. Used for the per-block scales. The NaN
// codes decode to 0 so a corrupt payload cannot inject NaNs.

struct e4m3_table {
    float val[256];
    e4m3_table() {
        for (int c = 0; c < 256; ++c) {
            const int e = (c >> 3) & 0xf;
            const int m = c & 0x7;
            float v = e ? ldexpf(1.0f + m/8.0f, e - 7) : ldexpf((float) m, -9);
            if ((c & 0x7f) == 0x7f) {
                v = 0.0f;
            }
            val[c] = (c & 0x80) ? -v : v;
        }
    }
};
static const e4m3_table E4M3;
static constexpr int E4M3_MAX_CODE = 0x7e;   // 448

// smallest positive code whose value is >= x (saturates at 448)
static inline int e4m3_ceil(float x) {
    const float * p = std::lower_bound(E4M3.val, E4M3.val + E4M3_MAX_CODE + 1, x);
    return std::min<int>((int) (p - E4M3.val), E4M3_MAX_CODE);
}

// nearest positive code to x
static inline int e4m3_nearest(float x) {
    const int c = e4m3_ceil(x);
    if (c > 0 && x - E4M3.val[c - 1] < E4M3.val[c] - x) {
        return c - 1;
    }
    return c;
}

// -------------------------------------------------------- lossy codecs ----

// the IQ4_NL grid (ggml-common.h kvalues_iq4nl): a non-uniform 16-level grid
// that fits the bell-shaped distribution of block-normalized values better
// than a uniform int4 or the fp4 e2m1 grid
static const float Q4_GRID[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// nearest grid index for t (in grid units). The midpoints between grid
// values all fall on multiples of 0.5, so a table over floor(2t) gives the
// exact nearest index (ties go up), without a search per value.
static constexpr int Q4_LUT_HALF = 256;   // covers t in [-128, 128)
static const struct q4_lut_t {
    uint8_t idx[2*Q4_LUT_HALF];
    q4_lut_t() {
        for (int b = 0; b < 2*Q4_LUT_HALF; ++b) {
            const float t = 0.5f*(b - Q4_LUT_HALF);   // lower edge of the bin
            int i = 0;
            while (i < 15 && 0.5f*(Q4_GRID[i] + Q4_GRID[i + 1]) <= t) {
                ++i;
            }
            idx[b] = (uint8_t) i;
        }
    }
} Q4_LUT;

static inline int q4_index(float t) {
    const int b = (int) floorf(2.0f*t) + Q4_LUT_HALF;
    return Q4_LUT.idx[std::max(0, std::min(2*Q4_LUT_HALF - 1, b))];
}

struct lossy_params {
    int  bits;      // bits per code
    int  block;     // values per block scale
    int  qmax;      // largest code magnitude (used for the row scale)
    bool grid;      // IQ4_NL grid instead of uniform signed integers
    int  search;    // block scale candidates tried below the ceiling (int) or around the nearest (grid)
};

static bool lossy_params_for(uint8_t codec, lossy_params & p) {
    switch (codec) {
        case RPC_CODEC_Q8: p = { 8, 32, 127, false, 0 }; return true;
        case RPC_CODEC_Q6: p = { 6, 32,  31, false, 1 }; return true;
        case RPC_CODEC_Q4: p = { 4, 16, 127, true,  2 }; return true;
        default:           return false;
    }
}

static uint8_t lossy_codec_for(uint8_t level) {
    switch (level) {
        case RPC_COMP_Q8: return RPC_CODEC_Q8;
        case RPC_COMP_Q6: return RPC_CODEC_Q6;
        case RPC_COMP_Q4: return RPC_CODEC_Q4;
        default:          return RPC_CODEC_RAW;
    }
}

// payload header of the lossy codecs: | row_len (u32) | n_rows (u32) | k (u16) | flags (u8) | reserved (u8) |
static constexpr size_t  LOSSY_HDR = 12;
static constexpr uint8_t LOSSY_FLAG_F32_OUTLIERS = 0x01;
// outliers sent exactly per row: one per 256 values, at least 4 (a model
// typically has a handful of massive channels whatever its width)
static constexpr uint32_t LOSSY_OUTLIER_DIV = 256;
static constexpr uint32_t LOSSY_OUTLIER_MIN = 4;

struct lossy_layout {
    size_t k;        // outliers per row
    size_t vbytes;   // bytes per outlier value (2: f16, 4: f32)
    size_t nb;       // blocks per row
    size_t code_bytes;
    size_t row_bytes;
};

static lossy_layout lossy_layout_for(const lossy_params & p, size_t row_len, size_t k, bool f32_outliers) {
    lossy_layout l;
    l.k          = k;
    l.vbytes     = f32_outliers ? 4 : 2;
    l.nb         = (row_len + p.block - 1) / p.block;
    l.code_bytes = (row_len * p.bits + 7) / 8;
    l.row_bytes  = sizeof(float) + k*(sizeof(uint16_t) + l.vbytes) + l.nb + l.code_bytes;
    return l;
}

// LSB-first bit packing of n codes of `bits` bits each
static void pack_codes(uint8_t * dst, const uint8_t * codes, size_t n, int bits) {
    if (bits == 8) {
        memcpy(dst, codes, n);
        return;
    }
    uint32_t acc = 0;
    int      nacc = 0;
    for (size_t i = 0; i < n; ++i) {
        acc |= (uint32_t) codes[i] << nacc;
        nacc += bits;
        while (nacc >= 8) {
            *dst++ = (uint8_t) acc;
            acc >>= 8;
            nacc -= 8;
        }
    }
    if (nacc > 0) {
        *dst = (uint8_t) acc;
    }
}

static void unpack_codes(uint8_t * codes, const uint8_t * src, size_t n, int bits) {
    if (bits == 8) {
        memcpy(codes, src, n);
        return;
    }
    const uint32_t mask = (1u << bits) - 1;
    uint32_t acc = 0;
    int      nacc = 0;
    for (size_t i = 0; i < n; ++i) {
        while (nacc < bits) {
            acc |= (uint32_t) *src++ << nacc;
            nacc += 8;
        }
        codes[i] = (uint8_t) (acc & mask);
        acc >>= bits;
        nacc -= bits;
    }
}

// quantizes one block with the scale s; returns the squared error and, if
// codes != nullptr, writes the codes
static float quantize_block(const lossy_params & p, const float * x, int n, float s, uint8_t * codes) {
    float err = 0.0f;
    if (s == 0.0f) {
        for (int i = 0; i < n; ++i) {
            err += x[i]*x[i];
            if (codes) {
                codes[i] = p.grid ? (uint8_t) q4_index(0.0f) : (uint8_t) (1 << (p.bits - 1));
            }
        }
        return err;
    }
    const float inv = 1.0f / s;
    for (int i = 0; i < n; ++i) {
        float r;
        int   c;
        if (p.grid) {
            c = q4_index(x[i]*inv);
            r = Q4_GRID[c]*s;
        } else {
            const int q = std::max(-p.qmax, std::min(p.qmax, (int) lrintf(x[i]*inv)));
            c = q + (1 << (p.bits - 1));
            r = (float) q*s;
        }
        err += (r - x[i])*(r - x[i]);
        if (codes) {
            codes[i] = (uint8_t) c;
        }
    }
    return err;
}

// encodes one row; `w` is scratch (row_len floats), `idx` scratch (row_len
// ints), `codes` scratch (row_len bytes)
static void encode_row(const lossy_params & p, const lossy_layout & l, const float * x, size_t row_len,
                       uint8_t * out, float * w, uint32_t * idx, uint8_t * codes) {
    memcpy(w, x, row_len*sizeof(float));

    // 1. the k largest magnitudes are sent exactly and removed from the row
    uint8_t * p_idx = out + sizeof(float);
    uint8_t * p_val = p_idx + l.k*sizeof(uint16_t);
    if (l.k > 0) {
        std::iota(idx, idx + row_len, 0u);
        std::nth_element(idx, idx + l.k - 1, idx + row_len,
                         [w](uint32_t a, uint32_t b) { return fabsf(w[a]) > fabsf(w[b]); });
        std::sort(idx, idx + l.k);
        for (size_t j = 0; j < l.k; ++j) {
            const uint16_t i16 = (uint16_t) idx[j];
            memcpy(p_idx + j*sizeof(uint16_t), &i16, sizeof(i16));
            if (l.vbytes == 4) {
                memcpy(p_val + j*4, &w[idx[j]], 4);
            } else {
                const ggml_fp16_t h = GGML_FP32_TO_FP16(w[idx[j]]);
                memcpy(p_val + j*2, &h, 2);
            }
            w[idx[j]] = 0.0f;
        }
    }

    // 2. row scale: the largest remaining magnitude maps to the top code at
    // the top block scale (448)
    float amax = 0.0f;
    for (size_t i = 0; i < row_len; ++i) {
        amax = std::max(amax, fabsf(w[i]));
    }
    const float g = amax / (p.qmax * 448.0f);
    memcpy(out, &g, sizeof(float));

    // 3. blocks
    uint8_t * p_scales = p_val + l.k*l.vbytes;
    uint8_t * p_codes  = p_scales + l.nb;
    for (size_t b = 0; b < l.nb; ++b) {
        const float * xb = w + b*p.block;
        const int     n  = (int) std::min<size_t>(p.block, row_len - b*p.block);
        float bmax = 0.0f;
        float bext = 0.0f;   // signed value of the largest magnitude
        for (int i = 0; i < n; ++i) {
            if (fabsf(xb[i]) > bmax) {
                bmax = fabsf(xb[i]);
                bext = xb[i];
            }
        }
        int best = 0;
        if (g > 0.0f && bmax > 0.0f) {
            float best_err = INFINITY;
            auto consider = [&](int code) {
                const float e = quantize_block(p, xb, n, E4M3.val[code]*g, nullptr);
                if (e < best_err) {
                    best_err = e;
                    best = code;
                }
            };
            if (!p.grid) {
                // ceiling scale never clips; one step lower trades a clipped
                // maximum for finer steps everywhere else
                const int c = e4m3_ceil(bmax / p.qmax / g);
                for (int k = 0; k <= p.search; ++k) {
                    if (c - k >= 1) {
                        consider(c - k);
                    }
                }
            } else {
                // the grid is asymmetric (-127..113), so try both signs: a
                // negative scale maps the extreme onto -127
                const int cp = e4m3_nearest(bmax / 127.0f / g);
                for (int sgn = 0; sgn < 2; ++sgn) {
                    for (int k = -p.search; k <= p.search; ++k) {
                        const int c = cp + k;
                        if (c >= 1 && c <= E4M3_MAX_CODE) {
                            consider(((bext > 0) == (sgn == 1)) ? (c | 0x80) : c);
                        }
                    }
                }
            }
        }
        p_scales[b] = (uint8_t) best;
        quantize_block(p, xb, n, E4M3.val[best]*g, codes + b*p.block);
    }
    pack_codes(p_codes, codes, row_len, p.bits);
}

static bool decode_row(const lossy_params & p, const lossy_layout & l, const uint8_t * in, size_t row_len,
                       float * y, uint8_t * codes) {
    float g;
    memcpy(&g, in, sizeof(float));
    const uint8_t * p_idx    = in + sizeof(float);
    const uint8_t * p_val    = p_idx + l.k*sizeof(uint16_t);
    const uint8_t * p_scales = p_val + l.k*l.vbytes;
    const uint8_t * p_codes  = p_scales + l.nb;

    unpack_codes(codes, p_codes, row_len, p.bits);
    const int offs = 1 << (p.bits - 1);
    for (size_t b = 0; b < l.nb; ++b) {
        const float s  = E4M3.val[p_scales[b]]*g;
        const size_t i0 = b*p.block;
        const size_t i1 = std::min<size_t>(i0 + p.block, row_len);
        if (p.grid) {
            for (size_t i = i0; i < i1; ++i) {
                y[i] = Q4_GRID[codes[i] & 0xf]*s;
            }
        } else {
            for (size_t i = i0; i < i1; ++i) {
                y[i] = (float) ((int) codes[i] - offs)*s;
            }
        }
    }
    for (size_t j = 0; j < l.k; ++j) {
        uint16_t i16;
        memcpy(&i16, p_idx + j*sizeof(uint16_t), sizeof(i16));
        if (i16 >= row_len) {
            return false;
        }
        if (l.vbytes == 4) {
            memcpy(&y[i16], p_val + j*4, 4);
        } else {
            ggml_fp16_t h;
            memcpy(&h, p_val + j*2, 2);
            y[i16] = GGML_FP16_TO_FP32(h);
        }
    }
    return true;
}

// appends the lossy encoding of n_rows x row_len floats to out. Returns false
// if the payload cannot be encoded (non-finite values).
static bool lossy_encode(std::vector<uint8_t> & out, uint8_t codec, const float * x, size_t row_len, size_t n_rows) {
    lossy_params p;
    if (!lossy_params_for(codec, p)) {
        return false;
    }
    // non-finite values (attention masks use -inf) have no code: the caller
    // falls back to the lossless codec, which also compresses masks far better
    float amax = 0.0f;
    const size_t n = row_len*n_rows;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i])) {
            return false;
        }
        amax = std::max(amax, fabsf(x[i]));
    }
    const bool   f32_outliers = amax > 65504.0f;
    const size_t k = row_len <= 65536 ? std::max<size_t>(LOSSY_OUTLIER_MIN, row_len / LOSSY_OUTLIER_DIV) : 0;
    const lossy_layout l = lossy_layout_for(p, row_len, k, f32_outliers);

    const size_t off = out.size();
    out.resize(off + LOSSY_HDR + n_rows*l.row_bytes);
    uint8_t * hdr = out.data() + off;
    const uint32_t row_len32 = (uint32_t) row_len;
    const uint32_t n_rows32  = (uint32_t) n_rows;
    const uint16_t k16       = (uint16_t) k;
    memcpy(hdr + 0, &row_len32, 4);
    memcpy(hdr + 4, &n_rows32,  4);
    memcpy(hdr + 8, &k16,       2);
    hdr[10] = f32_outliers ? LOSSY_FLAG_F32_OUTLIERS : 0;
    hdr[11] = 0;

    uint8_t * rows = hdr + LOSSY_HDR;
    parallel_rows(n_rows, row_len, [&](size_t r0, size_t r1) {
        std::vector<float>    w(row_len);
        std::vector<uint32_t> idx(row_len);
        std::vector<uint8_t>  codes(l.nb*p.block);
        for (size_t r = r0; r < r1; ++r) {
            encode_row(p, l, x + r*row_len, row_len, rows + r*l.row_bytes, w.data(), idx.data(), codes.data());
        }
    });
    return true;
}

static bool lossy_decode(float * y, size_t n, uint8_t codec, const uint8_t * in, size_t in_size) {
    lossy_params p;
    if (!lossy_params_for(codec, p) || in_size < LOSSY_HDR) {
        return false;
    }
    uint32_t row_len, n_rows;
    uint16_t k;
    memcpy(&row_len, in + 0, 4);
    memcpy(&n_rows,  in + 4, 4);
    memcpy(&k,       in + 8, 2);
    const uint8_t flags = in[10];
    if (row_len == 0 || (uint64_t) row_len*n_rows != n || k > row_len || (flags & ~LOSSY_FLAG_F32_OUTLIERS) != 0) {
        return false;
    }
    const lossy_layout l = lossy_layout_for(p, row_len, k, (flags & LOSSY_FLAG_F32_OUTLIERS) != 0);
    if (in_size - LOSSY_HDR != (uint64_t) n_rows*l.row_bytes) {
        return false;
    }
    const uint8_t * rows = in + LOSSY_HDR;
    bool ok = true;
    std::vector<uint8_t> chunk_ok(n_rows, 1);
    parallel_rows(n_rows, row_len, [&](size_t r0, size_t r1) {
        std::vector<uint8_t> codes(row_len);
        for (size_t r = r0; r < r1; ++r) {
            chunk_ok[r] = decode_row(p, l, rows + r*l.row_bytes, row_len, y + r*row_len, codes.data());
        }
    });
    for (uint8_t c : chunk_ok) {
        ok = ok && c;
    }
    return ok;
}


// ------------------------------------------------------- high level ----

static void write_header(std::vector<uint8_t> & out, uint8_t codec, uint64_t orig_size) {
    out[0] = codec;
    memcpy(out.data() + 1, &orig_size, sizeof(orig_size));
}

uint8_t rpc_compress_payload(std::vector<uint8_t> & out, const void * data, size_t size,
                             enum ggml_type type, uint32_t row_len, uint8_t level) {
    if (level == RPC_COMP_NONE || !rpc_compress_level_valid(level) || size == 0) {
        return RPC_CODEC_RAW;
    }
    std::vector<uint8_t> buf(RPC_COMP_HEADER_SIZE);
    const size_t es = float_elem_size(type);

    // lossy path: whole rows of a float type, all values finite
    if (rpc_compress_level_lossy(level) && es != 0 && row_len >= RPC_COMP_MIN_ROW && size % (row_len*es) == 0) {
        const size_t n = size / es;
        std::vector<float> f32;
        const float * x = (const float *) data;
        if (type != GGML_TYPE_F32 || ((uintptr_t) data % alignof(float)) != 0) {
            f32.resize(n);
            to_f32(data, f32.data(), n, type);
            x = f32.data();
        }
        uint8_t codec = lossy_codec_for(level);
        if (lossy_encode(buf, codec, x, row_len, n / row_len)) {
            std::vector<uint8_t> wrapped(RPC_COMP_HEADER_SIZE);
            if (zstd_compress_append(wrapped, buf.data() + RPC_COMP_HEADER_SIZE, buf.size() - RPC_COMP_HEADER_SIZE, ZSTD_LEVEL_FLOAT) &&
                wrapped.size() < buf.size()) {
                buf.swap(wrapped);
                codec |= RPC_CODEC_FLAG_ZSTD;
            }
            write_header(buf, codec, size);
            out.swap(buf);
            return codec;
        }
        buf.resize(RPC_COMP_HEADER_SIZE);
    }

    // lossless path, kept only if it actually shrinks the payload
    uint8_t codec = RPC_CODEC_RAW;
    if (es > 1 && size % es == 0) {
        std::vector<uint8_t> shuffled(size);
        shuffle_bytes(shuffled.data(), (const uint8_t *) data, size / es, es);
        buf.push_back((uint8_t) es);
        if (zstd_compress_append(buf, shuffled.data(), size, ZSTD_LEVEL_FLOAT)) {
            codec = RPC_CODEC_ZSTD_SHUF;
        }
    } else if (zstd_compress_append(buf, data, size, ZSTD_LEVEL_OTHER)) {
        codec = RPC_CODEC_ZSTD;
    }
    if (codec == RPC_CODEC_RAW || buf.size() >= size) {
        return RPC_CODEC_RAW;
    }
    write_header(buf, codec, size);
    out.swap(buf);
    return codec;
}

bool rpc_compressed_orig_size(const void * data, size_t size, uint64_t & orig_size) {
    if (size < RPC_COMP_HEADER_SIZE) {
        return false;
    }
    memcpy(&orig_size, (const uint8_t *) data + 1, sizeof(orig_size));
    return true;
}

bool rpc_decompress_payload(void * out, size_t out_size, const void * data, size_t size,
                            enum ggml_type type) {
    uint64_t orig_size;
    if (!rpc_compressed_orig_size(data, size, orig_size) || out_size == 0) {
        return false;
    }
    // the receiver always knows how big the payload should be, so a corrupt
    // or hostile header can never make us write out of bounds
    if (orig_size != out_size) {
        LOG_ERROR("[%s] payload size mismatch: got %" PRIu64 ", expected %zu\n", __func__, orig_size, out_size);
        return false;
    }
    const uint8_t   codec        = *(const uint8_t *) data;
    const uint8_t * payload      = (const uint8_t *) data + RPC_COMP_HEADER_SIZE;
    const size_t    payload_size = size - RPC_COMP_HEADER_SIZE;

    switch (codec) {
        case RPC_CODEC_RAW:
            if (payload_size != out_size) {
                return false;
            }
            memcpy(out, payload, out_size);
            return true;
        case RPC_CODEC_ZSTD:
            return zstd_decompress(out, out_size, payload, payload_size);
        case RPC_CODEC_ZSTD_SHUF: {
            if (payload_size < 1) {
                return false;
            }
            const size_t es = payload[0];
            if (es < 2 || es > 8 || out_size % es != 0) {
                return false;
            }
            std::vector<uint8_t> shuffled(out_size);
            if (!zstd_decompress(shuffled.data(), out_size, payload + 1, payload_size - 1)) {
                return false;
            }
            unshuffle_bytes((uint8_t *) out, shuffled.data(), out_size / es, es);
            return true;
        }
        case RPC_CODEC_Q8:
        case RPC_CODEC_Q6:
        case RPC_CODEC_Q4:
        case RPC_CODEC_Q8 | RPC_CODEC_FLAG_ZSTD:
        case RPC_CODEC_Q6 | RPC_CODEC_FLAG_ZSTD:
        case RPC_CODEC_Q4 | RPC_CODEC_FLAG_ZSTD: {
            // reconstructed in the tensor's native float type, which both
            // sides know from the rpc_tensor header
            const size_t es = float_elem_size(type);
            if (es == 0 || out_size % es != 0) {
                LOG_ERROR("[%s] lossy payload for non-float type %s\n", __func__, ggml_type_name(type));
                return false;
            }
            const size_t n = out_size / es;
            const uint8_t * stream      = payload;
            size_t          stream_size = payload_size;
            std::vector<uint8_t> unwrapped;
            if (codec & RPC_CODEC_FLAG_ZSTD) {
                // a lossy stream is never larger than the plain payload (it
                // has fewer bits per value), which bounds the allocation
                if (!zstd_decompress_bounded(unwrapped, payload, payload_size, out_size + LOSSY_HDR)) {
                    return false;
                }
                stream      = unwrapped.data();
                stream_size = unwrapped.size();
            }
            const uint8_t base = codec & ~RPC_CODEC_FLAG_ZSTD;
            if (type == GGML_TYPE_F32 && ((uintptr_t) out % alignof(float)) == 0) {
                return lossy_decode((float *) out, n, base, stream, stream_size);
            }
            std::vector<float> f32(n);
            if (!lossy_decode(f32.data(), n, base, stream, stream_size)) {
                return false;
            }
            from_f32(f32.data(), out, n, type);
            return true;
        }
        default:
            LOG_ERROR("[%s] unknown codec %u\n", __func__, codec);
            return false;
    }
}
