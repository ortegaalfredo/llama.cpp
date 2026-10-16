// Unit tests for the RPC transport codecs (ggml/src/ggml-rpc/rpc-compress.cpp):
// lossless round trips, the accuracy of the lossy levels on activation-like
// data with massive outlier channels, fallbacks, and robustness against
// corrupt payloads.
#include "ggml.h"
#include "ggml-rpc/rpc-compress.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_failed = 0;

#define CHECK(cond, ...) do {                         \
    if (!(cond)) {                                    \
        printf("FAIL %s:%d: ", __FILE__, __LINE__);   \
        printf(__VA_ARGS__);                          \
        printf("\n");                                 \
        n_failed++;                                   \
    }                                                 \
} while (0)

// activation-like rows: gaussian bulk plus a few massive channels at fixed
// positions, like the residual stream of a real LLM
static std::vector<float> make_activations(size_t row_len, size_t n_rows, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(row_len*n_rows);
    for (size_t r = 0; r < n_rows; ++r) {
        for (size_t i = 0; i < row_len; ++i) {
            x[r*row_len + i] = nd(rng) * (0.5f + (i % 7)*0.2f);
        }
        x[r*row_len + 3 % row_len]   = 300.0f + r;
        x[r*row_len + 35 % row_len]  = -150.0f;
        x[r*row_len + 911 % row_len] = 2000.0f;
    }
    return x;
}

// relative error, averaged per row, over the values below the 99th
// percentile of each row (the bulk that carries the information)
static double bulk_rel_err(const std::vector<float> & x, const std::vector<float> & y, size_t row_len) {
    double sum = 0.0;
    const size_t n_rows = x.size() / row_len;
    for (size_t r = 0; r < n_rows; ++r) {
        std::vector<float> a(row_len);
        for (size_t i = 0; i < row_len; ++i) {
            a[i] = fabsf(x[r*row_len + i]);
        }
        std::nth_element(a.begin(), a.begin() + row_len*99/100, a.end());
        const float thr = a[row_len*99/100];
        double e = 0.0, s = 0.0;
        for (size_t i = 0; i < row_len; ++i) {
            const float v = x[r*row_len + i];
            if (fabsf(v) < thr) {
                e += (double) (y[r*row_len + i] - v)*(y[r*row_len + i] - v);
                s += (double) v*v;
            }
        }
        sum += sqrt(e / s);
    }
    return sum / n_rows;
}

static bool round_trip(const void * data, size_t size, ggml_type type, uint32_t row_len, uint8_t level,
                       std::vector<uint8_t> & out, uint8_t & codec, size_t & wire_size) {
    std::vector<uint8_t> wire;
    codec = rpc_compress_payload(wire, data, size, type, row_len, level);
    out.assign(size, 0xAA);
    if (codec == RPC_CODEC_RAW) {
        wire_size = size;
        memcpy(out.data(), data, size);
        return true;
    }
    wire_size = wire.size();
    return rpc_decompress_payload(out.data(), size, wire.data(), wire.size(), type);
}

static void test_lossless() {
    std::mt19937 rng(1);
    const auto acts = make_activations(1024, 16, 2);

    // float payloads at the lossless level, and lossy levels with a row
    // length that makes them ineligible, must round trip bit-exactly
    struct { uint8_t level; uint32_t row_len; } cases[] = {
        { RPC_COMP_ZSTD, 1024 }, { RPC_COMP_Q4, 0 }, { RPC_COMP_Q8, 128 }, { RPC_COMP_Q6, 1000 },
    };
    for (const auto & c : cases) {
        std::vector<uint8_t> out;
        uint8_t codec;
        size_t  wire_size;
        bool ok = round_trip(acts.data(), acts.size()*4, GGML_TYPE_F32, c.row_len, c.level, out, codec, wire_size);
        CHECK(ok && memcmp(out.data(), acts.data(), out.size()) == 0, "f32 lossless round trip (level %s, row %u)",
              rpc_compress_level_name(c.level), c.row_len);
        CHECK(codec == RPC_CODEC_ZSTD_SHUF || codec == RPC_CODEC_RAW, "f32 at a lossless level used codec %u", codec);
    }

    // f16 / bf16
    for (ggml_type type : { GGML_TYPE_F16, GGML_TYPE_BF16 }) {
        std::vector<uint16_t> h(acts.size());
        for (size_t i = 0; i < acts.size(); ++i) {
            h[i] = type == GGML_TYPE_F16 ? ggml_fp32_to_fp16(acts[i]) : ggml_fp32_to_bf16(acts[i]).bits;
        }
        std::vector<uint8_t> out;
        uint8_t codec;
        size_t  wire_size;
        bool ok = round_trip(h.data(), h.size()*2, type, 1024, RPC_COMP_ZSTD, out, codec, wire_size);
        CHECK(ok && memcmp(out.data(), h.data(), out.size()) == 0, "%s lossless round trip", ggml_type_name(type));
    }

    // non-float payloads (quantized weights, token ids) never take a lossy codec
    std::vector<uint8_t> q(18*4096);
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = (uint8_t) (rng() % 16 * (i % 3 == 0));   // compressible
    }
    for (uint8_t level = RPC_COMP_ZSTD; level < RPC_COMP_COUNT; ++level) {
        std::vector<uint8_t> out;
        uint8_t codec;
        size_t  wire_size;
        bool ok = round_trip(q.data(), q.size(), GGML_TYPE_Q4_0, 32, level, out, codec, wire_size);
        CHECK(ok && out == q, "q4_0 payload round trip at level %s", rpc_compress_level_name(level));
        CHECK(codec == RPC_CODEC_ZSTD && wire_size < q.size(), "q4_0 payload codec %u, %zu -> %zu", codec, q.size(), wire_size);
    }

    // incompressible data is sent raw
    std::vector<uint8_t> noise(65536);
    for (auto & b : noise) {
        b = (uint8_t) rng();
    }
    std::vector<uint8_t> wire;
    CHECK(rpc_compress_payload(wire, noise.data(), noise.size(), GGML_TYPE_I8, 0, RPC_COMP_Q4) == RPC_CODEC_RAW,
          "incompressible payload not sent raw");
}

static void test_lossy_accuracy() {
    struct { uint8_t level; uint8_t codec; double max_bulk_err; double min_ratio; } levels[] = {
        { RPC_COMP_Q8, RPC_CODEC_Q8, 0.010, 3.7 },
        { RPC_COMP_Q6, RPC_CODEC_Q6, 0.035, 4.9 },
        { RPC_COMP_Q4, RPC_CODEC_Q4, 0.100, 6.7 },
    };
    for (size_t row_len : { (size_t) 1024, (size_t) 4096, (size_t) 300 }) {
        const size_t n_rows = 64;
        const auto x = make_activations(row_len, n_rows, 7);
        for (const auto & l : levels) {
            std::vector<uint8_t> out;
            uint8_t codec;
            size_t  wire_size;
            bool ok = round_trip(x.data(), x.size()*4, GGML_TYPE_F32, (uint32_t) row_len, l.level, out, codec, wire_size);
            CHECK(ok, "lossy decode failed (level %s, row %zu)", rpc_compress_level_name(l.level), row_len);
            CHECK((codec & ~RPC_CODEC_FLAG_ZSTD) == l.codec, "level %s used codec %u", rpc_compress_level_name(l.level), codec);
            std::vector<float> y(x.size());
            memcpy(y.data(), out.data(), out.size());
            const double err   = bulk_rel_err(x, y, row_len);
            const double ratio = (double) (x.size()*4) / wire_size;
            printf("  %-3s row %4zu: bulk rel err %.4f, ratio %.2fx\n", rpc_compress_level_name(l.level), row_len, err, ratio);
            CHECK(err < l.max_bulk_err, "level %s row %zu: bulk error %.4f >= %.4f", rpc_compress_level_name(l.level), row_len, err, l.max_bulk_err);
            if (row_len >= 1024) {
                CHECK(ratio > l.min_ratio, "level %s row %zu: ratio %.2f <= %.2f", rpc_compress_level_name(l.level), row_len, ratio, l.min_ratio);
            }
            // the massive channels are sent exactly (to f16 precision)
            for (size_t r = 0; r < n_rows; ++r) {
                const size_t i = r*row_len + 911 % row_len;
                CHECK(fabsf(y[i] - x[i]) <= 1.0f, "outlier not preserved: %f -> %f", x[i], y[i]);
            }
        }
    }

    // f16 input reconstructs as f16
    const auto x = make_activations(1024, 8, 3);
    std::vector<ggml_fp16_t> h(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        h[i] = ggml_fp32_to_fp16(x[i]);
    }
    std::vector<uint8_t> out;
    uint8_t codec;
    size_t  wire_size;
    bool ok = round_trip(h.data(), h.size()*2, GGML_TYPE_F16, 1024, RPC_COMP_Q8, out, codec, wire_size);
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = ggml_fp16_to_fp32(((ggml_fp16_t *) out.data())[i]);
    }
    CHECK(ok && (codec & ~RPC_CODEC_FLAG_ZSTD) == RPC_CODEC_Q8 && bulk_rel_err(x, y, 1024) < 0.01, "f16 q8 round trip");

    // an all-zero payload (unused graph inputs, a mask with nothing masked)
    // is exact and collapses to almost nothing
    std::vector<float> zeros(4096*512, 0.0f);
    for (uint8_t level : { (uint8_t) RPC_COMP_Q8, (uint8_t) RPC_COMP_Q6, (uint8_t) RPC_COMP_Q4 }) {
        ok = round_trip(zeros.data(), zeros.size()*4, GGML_TYPE_F32, 4096, level, out, codec, wire_size);
        CHECK(ok && memcmp(out.data(), zeros.data(), out.size()) == 0, "zero payload not exact");
        CHECK(wire_size*1000 < zeros.size()*4, "zero payload at %s: %zu -> %zu bytes", rpc_compress_level_name(level), zeros.size()*4, wire_size);
    }

    // values beyond the f16 range survive as f32 outliers
    auto big = make_activations(1024, 4, 5);
    big[10] = 1e6f;
    big[1024 + 20] = -3e7f;
    ok = round_trip(big.data(), big.size()*4, GGML_TYPE_F32, 1024, RPC_COMP_Q4, out, codec, wire_size);
    std::vector<float> yb(big.size());
    memcpy(yb.data(), out.data(), out.size());
    CHECK(ok && yb[10] == 1e6f && yb[1024 + 20] == -3e7f, "huge outliers not preserved");
}

static void test_non_finite_fallback() {
    // attention masks: 0 / -inf. They must survive exactly, so a lossy level
    // falls back to the lossless codec for them
    std::vector<float> mask(512*256);
    for (size_t r = 0; r < 256; ++r) {
        for (size_t c = 0; c < 512; ++c) {
            mask[r*512 + c] = c <= r ? 0.0f : -INFINITY;
        }
    }
    std::vector<uint8_t> out;
    uint8_t codec;
    size_t  wire_size;
    bool ok = round_trip(mask.data(), mask.size()*4, GGML_TYPE_F32, 512, RPC_COMP_Q4, out, codec, wire_size);
    CHECK(ok && memcmp(out.data(), mask.data(), out.size()) == 0, "mask not exact");
    CHECK(codec == RPC_CODEC_ZSTD_SHUF && wire_size*20 < mask.size()*4, "mask codec %u, %zu bytes", codec, wire_size);

    auto x = make_activations(1024, 2, 9);
    x[5] = NAN;
    ok = round_trip(x.data(), x.size()*4, GGML_TYPE_F32, 1024, RPC_COMP_Q8, out, codec, wire_size);
    CHECK(ok && memcmp(out.data(), x.data(), out.size()) == 0 && codec != RPC_CODEC_Q8, "NaN payload not lossless");
}

static void test_corrupt_payloads() {
    const auto x = make_activations(1024, 8, 11);
    std::mt19937 rng(12);
    for (uint8_t level : { (uint8_t) RPC_COMP_ZSTD, (uint8_t) RPC_COMP_Q8, (uint8_t) RPC_COMP_Q6, (uint8_t) RPC_COMP_Q4 }) {
        std::vector<uint8_t> wire;
        rpc_compress_payload(wire, x.data(), x.size()*4, GGML_TYPE_F32, 1024, level);
        std::vector<float> y(x.size());

        // wrong expected size, wrong type, truncation: always rejected
        CHECK(!rpc_decompress_payload(y.data(), x.size()*4 - 4, wire.data(), wire.size(), GGML_TYPE_F32), "size mismatch accepted");
        for (size_t cut : { (size_t) 0, (size_t) 5, (size_t) 9, (size_t) 20, wire.size()/2, wire.size() - 1 }) {
            CHECK(!rpc_decompress_payload(y.data(), x.size()*4, wire.data(), cut, GGML_TYPE_F32), "truncated payload accepted (%zu)", cut);
        }
        if (level != RPC_COMP_ZSTD) {
            CHECK(!rpc_decompress_payload(y.data(), x.size()*4, wire.data(), wire.size(), GGML_TYPE_I32), "lossy payload accepted for i32");
            // an outlier index past the row end is rejected
            std::vector<uint8_t> bad = wire;
            const size_t first_idx = RPC_COMP_HEADER_SIZE + 12 + 4;
            const uint16_t idx = 1024;
            memcpy(bad.data() + first_idx, &idx, 2);
            CHECK(!rpc_decompress_payload(y.data(), x.size()*4, bad.data(), bad.size(), GGML_TYPE_F32), "outlier index out of range accepted");
        }
        // random corruption: may decode to garbage, but must never crash or
        // write out of bounds (run under ASan to check the latter). The
        // rejections are logged as errors, so mute the log meanwhile.
        ggml_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
        for (int it = 0; it < 2000; ++it) {
            std::vector<uint8_t> bad = wire;
            const int n_flips = 1 + rng() % 8;
            for (int f = 0; f < n_flips; ++f) {
                const size_t pos = rng() % std::min<size_t>(bad.size(), it % 2 ? bad.size() : 64);
                bad[pos] ^= (uint8_t) (1u << (rng() % 8));
            }
            rpc_decompress_payload(y.data(), x.size()*4, bad.data(), bad.size(), GGML_TYPE_F32);
        }
        ggml_log_set(nullptr, nullptr);
    }
}

int main() {
    printf("lossless:\n");
    test_lossless();
    printf("lossy accuracy:\n");
    test_lossy_accuracy();
    printf("non-finite fallback:\n");
    test_non_finite_fallback();
    printf("corrupt payloads:\n");
    test_corrupt_payloads();
    printf("%s\n", n_failed ? "FAILED" : "ALL PASSED");
    return n_failed ? 1 : 0;
}
