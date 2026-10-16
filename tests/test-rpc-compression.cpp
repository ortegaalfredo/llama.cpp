// End-to-end test for RPC transport compression against a real rpc server.
// Checks the payload policy for the negotiated level:
//  - weight buffers and buffers with usage ANY (the KV cache) are exact
//  - compute-buffer activations are quantized at lossy levels, within the
//    level's error, and exact otherwise
//  - reads of tensors flagged as graph outputs (logits, embeddings) are exact
//  - attention masks (-inf) are exact
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int n_failed = 0;

#define CHECK(cond, ...) do {                         \
    printf("%s: ", (cond) ? "ok" : "FAIL");           \
    printf(__VA_ARGS__);                              \
    printf("\n");                                     \
    if (!(cond)) {                                    \
        n_failed++;                                   \
    }                                                 \
} while (0)

static const int64_t ROW = 2048;
static const int64_t ROWS = 32;

// relative error, averaged per row, over the values below the 99th
// percentile of each row
static double bulk_rel_err(const std::vector<float> & x, const std::vector<float> & y) {
    double sum = 0.0;
    for (int64_t r = 0; r < ROWS; ++r) {
        std::vector<float> a(ROW);
        for (int64_t i = 0; i < ROW; ++i) {
            a[i] = fabsf(x[r*ROW + i]);
        }
        std::nth_element(a.begin(), a.begin() + ROW*99/100, a.end());
        const float thr = a[ROW*99/100];
        double e = 0.0, s = 0.0;
        for (int64_t i = 0; i < ROW; ++i) {
            const float v = x[r*ROW + i];
            if (fabsf(v) < thr) {
                e += (double) (y[r*ROW + i] - v)*(y[r*ROW + i] - v);
                s += (double) v*v;
            }
        }
        sum += sqrt(e / s);
    }
    return sum / ROWS;
}

static bool lossy_level(const std::string & level) {
    return level == "q8" || level == "q6" || level == "q4";
}

// bulk error bound for one quantization hop at a level
static double hop_bound(const std::string & level) {
    if (level == "q8") return 0.012;
    if (level == "q6") return 0.045;
    if (level == "q4") return 0.11;
    return 0.0;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        printf("usage: %s <endpoint> <expected negotiated level>\n", argv[0]);
        return 1;
    }
    // the requested level comes from GGML_RPC_COMPRESSION, set by the test script
    ggml_backend_load_all();
    const char *      endpoint = argv[1];
    const std::string level    = argv[2];
    const bool        lossy    = lossy_level(level);

    ggml_backend_t backend = ggml_backend_rpc_init(endpoint, 0);
    if (backend == nullptr) {
        printf("failed to connect to %s\n", endpoint);
        return 1;
    }

    // activation-like data: gaussian bulk with massive channels
    std::vector<float> x(ROW*ROWS);
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (int64_t r = 0; r < ROWS; ++r) {
        for (int64_t i = 0; i < ROW; ++i) {
            x[r*ROW + i] = nd(rng);
        }
        x[r*ROW + 35]   = 1500.0f;
        x[r*ROW + 1000] = -400.0f;
    }
    const size_t nbytes = x.size()*sizeof(float);

    auto alloc = [&](ggml_context ** ctx, int n_tensors, ggml_type type, enum ggml_backend_buffer_usage usage,
                     std::vector<ggml_tensor *> & out) {
        ggml_init_params params = { (size_t) n_tensors*ggml_tensor_overhead(), nullptr, true };
        *ctx = ggml_init(params);
        for (int i = 0; i < n_tensors; ++i) {
            out.push_back(ggml_new_tensor_2d(*ctx, type, ROW, ROWS));
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(*ctx, backend);
        ggml_backend_buffer_set_usage(buf, usage);
        return buf;
    };

    std::vector<ggml_tensor *> w, a, kv;
    ggml_context * ctx_w, * ctx_a, * ctx_kv;
    ggml_backend_buffer_t buf_w  = alloc(&ctx_w,  1, GGML_TYPE_F32, GGML_BACKEND_BUFFER_USAGE_WEIGHTS, w);
    ggml_backend_buffer_t buf_a  = alloc(&ctx_a,  3, GGML_TYPE_F32, GGML_BACKEND_BUFFER_USAGE_COMPUTE, a);
    ggml_backend_buffer_t buf_kv = alloc(&ctx_kv, 1, GGML_TYPE_F32, GGML_BACKEND_BUFFER_USAGE_ANY,     kv);
    ggml_tensor * act = a[0];
    ggml_tensor * out = a[1];
    ggml_tensor * act2 = a[2];
    ggml_set_output(out);

    std::vector<float> y(x.size());

    // weights: exact both ways
    ggml_backend_tensor_set(w[0], x.data(), 0, nbytes);
    ggml_backend_tensor_get(w[0], y.data(), 0, nbytes);
    CHECK(memcmp(x.data(), y.data(), nbytes) == 0, "weights round trip exact");

    // KV cache (usage ANY): exact both ways, e.g. for state save / restore
    ggml_backend_tensor_set(kv[0], x.data(), 0, nbytes);
    ggml_backend_tensor_get(kv[0], y.data(), 0, nbytes);
    CHECK(memcmp(x.data(), y.data(), nbytes) == 0, "kv cache round trip exact");

    // put the exact data into the compute buffer with server-side copies
    ggml_backend_tensor_copy(w[0], act);
    ggml_backend_tensor_copy(w[0], out);

    // graph output: read back exactly, sync and async
    ggml_backend_tensor_get(out, y.data(), 0, nbytes);
    CHECK(memcmp(x.data(), y.data(), nbytes) == 0, "output read exact");
    std::fill(y.begin(), y.end(), 0.0f);
    ggml_backend_tensor_get_async(backend, out, y.data(), 0, nbytes);
    ggml_backend_synchronize(backend);
    CHECK(memcmp(x.data(), y.data(), nbytes) == 0, "output async read exact");

    // activation read: one quantization hop at lossy levels
    ggml_backend_tensor_get(act, y.data(), 0, nbytes);
    double err = bulk_rel_err(x, y);
    if (lossy) {
        CHECK(err > 0.0 && err < hop_bound(level), "activation read quantized, bulk err %.4f < %.3f", err, hop_bound(level));
        CHECK(fabsf(y[35] - 1500.0f) < 1.0f && fabsf(y[1000] + 400.0f) < 0.5f, "massive channels preserved (%.2f, %.2f)", y[35], y[1000]);
    } else {
        CHECK(err == 0.0, "activation read exact at level %s", level.c_str());
    }
    std::fill(y.begin(), y.end(), 0.0f);
    ggml_backend_tensor_get_async(backend, act, y.data(), 0, nbytes);
    ggml_backend_synchronize(backend);
    err = bulk_rel_err(x, y);
    CHECK(lossy ? (err > 0.0 && err < hop_bound(level)) : err == 0.0, "activation async read, bulk err %.4f", err);

    // activation write + read: two hops
    ggml_backend_tensor_set(act2, x.data(), 0, nbytes);
    ggml_backend_tensor_get(act2, y.data(), 0, nbytes);
    err = bulk_rel_err(x, y);
    CHECK(lossy ? (err > 0.0 && err < 1.5*hop_bound(level)) : err == 0.0, "activation write+read, bulk err %.4f", err);

    // 2D write + read of whole rows (two copies of 16 rows)
    std::fill(y.begin(), y.end(), 0.0f);
    const size_t half = nbytes/2;
    ggml_backend_tensor_set_2d(act2, x.data(), 0, half, 2, half, half);
    ggml_backend_tensor_get_2d(act2, y.data(), 0, half, 2, half, half);
    err = bulk_rel_err(x, y);
    CHECK(lossy ? (err > 0.0 && err < 1.5*hop_bound(level)) : err == 0.0, "activation 2d write+read, bulk err %.4f", err);
    ggml_backend_tensor_set_2d(w[0], x.data(), 0, half, 2, half, half);
    ggml_backend_tensor_get_2d(w[0], y.data(), 0, half, 2, half, half);
    CHECK(memcmp(x.data(), y.data(), nbytes) == 0, "weights 2d round trip exact");

    // attention mask with -inf in the compute buffer: exact
    std::vector<float> mask(x.size());
    for (int64_t r = 0; r < ROWS; ++r) {
        for (int64_t i = 0; i < ROW; ++i) {
            mask[r*ROW + i] = i <= r*64 ? 0.0f : -INFINITY;
        }
    }
    ggml_backend_tensor_set(act2, mask.data(), 0, nbytes);
    ggml_backend_tensor_get(act2, y.data(), 0, nbytes);
    CHECK(memcmp(mask.data(), y.data(), nbytes) == 0, "mask round trip exact");

    ggml_backend_buffer_free(buf_w);
    ggml_backend_buffer_free(buf_a);
    ggml_backend_buffer_free(buf_kv);
    ggml_free(ctx_w);
    ggml_free(ctx_a);
    ggml_free(ctx_kv);
    ggml_backend_free(backend);
    printf("%s\n", n_failed ? "FAILURES" : "ALL PASSED");
    return n_failed ? 1 : 0;
}
