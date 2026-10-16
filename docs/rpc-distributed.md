# Distributed inference over slow links: RPC transport compression

This build of llama.cpp compresses the data the RPC backend sends between the main host and the `ggml-rpc-server`
instances. It is meant for running one model across machines connected by an ordinary network - a home LAN, Wi-Fi,
or the internet - where bandwidth, not compute, limits throughput.

- **zstd** (lossless) - default. Weights and every non-float payload, plus float payloads after a byte shuffle.
- **q8 / q6 / q4** (lossy) - activations only, quantized to ~8.4 / ~6.4 / ~4.6 bits per value. Weights, the KV cache
  and the model's outputs (logits, embeddings) are never quantized.

At `q8` the activations crossing the network are 4x smaller than plain f32 with a KL divergence of ~4e-5 against an
uncompressed run - indistinguishable from sending them as fp16. At `q4` they are 7x smaller at a KLD of ~6e-3.
See [Measurements](#measurements).

When the model's output layer runs on a remote server, also pass **`-bs` (backend sampling)**: the next token is then
picked on the server and only its id crosses the network, instead of the full logits (~600 KB per token for a 151k
vocabulary). See [Remote output layer: sample on the server](#remote-output-layer-sample-on-the-server--bs).

For the basics of the RPC backend (building, devices, local cache, RDMA) see [tools/rpc/README.md](../tools/rpc/README.md).

> [!WARNING]
> The RPC protocol has **no authentication and no encryption**. Compression does not change that. Never expose
> `ggml-rpc-server` directly to the internet: put it behind an SSH tunnel, WireGuard, Tailscale or a similar VPN
> (see the [examples](#usage-examples)).

## Contents

- [Quick start](#quick-start)
- [Choosing a level](#choosing-a-level)
- [What gets compressed](#what-gets-compressed)
- [Usage examples](#usage-examples)
- [Estimating bandwidth](#estimating-bandwidth)
- [Measurements](#measurements)
- [How it works](#how-it-works)
- [Wire format](#wire-format)
- [Limitations and known issues](#limitations-and-known-issues)
- [Tests](#tests)

## Quick start

Build with RPC support on every machine. zstd is picked up automatically when its development package is installed
(`libzstd-dev` on Debian/Ubuntu, `zstd` on Homebrew); without it, compression is compiled out and the peer is told so
during the handshake.

```bash
sudo apt install libzstd-dev          # or: brew install zstd
cmake -B build -DGGML_RPC=ON -DGGML_CUDA=ON   # plus whatever backends the machine has
cmake --build build --config Release -j
```

On each remote machine:

```bash
build/bin/ggml-rpc-server --host 127.0.0.1 --port 50052
```

On the main host, pick the level with `GGML_RPC_COMPRESSION`:

```bash
GGML_RPC_COMPRESSION=q8 build/bin/llama-server -m model.gguf -ngl 99 --rpc 127.0.0.1:50052 -bs \
    --ctx-checkpoints 0 --cache-ram 0
```

`--ctx-checkpoints 0 --cache-ram 0` stop `llama-server` from copying the model's KV cache and recurrent state from the
servers to the main host after every request - up to ~200 MB per request; see
[Recommended llama-server settings](#recommended-llama-server-settings).

The server logs the level each connection negotiated:

```
client connected, transport: TCP, compression: q8
```

## Choosing a level

Each side states the most aggressive level it accepts; the connection uses the lower of the two. A server accepts every
level by default, so the client decides. `rpc-server -z LEVEL` caps what a server accepts (e.g. an operator who wants
to forbid lossy transfers runs `-z zstd`).

| Level  | Activations          | Activation size vs f32 | Quality cost (KLD, Qwen3-0.6B, 3 lossy hops) | Use when |
|--------|----------------------|------------------------|-----------------------------------------------|----------|
| `none` | raw                  | 1x                     | 0                                             | fast LAN (10 GbE+), or to rule compression out while debugging |
| `zstd` | lossless             | ~1.19x smaller         | 0 (bit-exact)                                 | default; always safe |
| `q8`   | int8, blocks of 32   | 4.0x smaller           | 0.00004                                       | **recommended for any link slower than ~1 Gbit/s** |
| `q6`   | int6, blocks of 32   | 5.1x smaller           | 0.0006                                        | thin links where q8 is still the bottleneck |
| `q4`   | IQ4_NL grid, blocks of 16 | 7.1x smaller      | 0.0063                                        | very thin links (mobile, congested WAN) |

For scale: quantizing the *weights* of a model from f16 to Q4_K_M typically costs a KLD of several hundredths, ten or
more times what `q4` transmission adds.

Level names: `none`, `zstd`, `q8`, `q6`, `q4`, or `0`-`4`. `fp8` and `fp4` are accepted as aliases of `q8` and `q4`
for scripts written for the earlier version of this feature.

## What gets compressed

Only the tensor data paths (`SET_TENSOR`, `SET_TENSOR_2D`, `GET_TENSOR`, `GET_TENSOR_2D`) are compressed; control
messages are a few hundred bytes and stay as they are. The client decides per payload:

| Payload | Codec |
|---------|-------|
| Weights (buffer usage `WEIGHTS`) | zstd, always lossless - sent once at load |
| KV cache (buffer usage `ANY`), e.g. state save/restore | lossless |
| Graph outputs read back by the application - logits, embeddings (tensors flagged `GGML_TENSOR_FLAG_OUTPUT`) | lossless |
| Integer inputs - token ids, positions, KV indices | lossless |
| Attention masks, or any float payload containing `inf`/`NaN` | lossless (masks compress ~100x anyway) |
| Float tensors narrower than 256 values per row (per-token scalars) | lossless |
| **Activations**: f32/f16/bf16 tensors in compute buffers - the hidden states the scheduler copies between machines | **lossy at q8/q6/q4** |

A lossless codec is only kept when it makes the payload smaller; otherwise the payload is sent raw.

## Usage examples

### Two machines over the internet through an SSH tunnel

The server binds to localhost only (the default), and the client reaches it through an encrypted tunnel.

On the remote machine (`gpu-box`):

```bash
build/bin/ggml-rpc-server --host 127.0.0.1 --port 50052
```

On the main host:

```bash
ssh -N -L 50052:127.0.0.1:50052 user@gpu-box &
GGML_RPC_COMPRESSION=q8 build/bin/llama-server -m Qwen3-8B-Q4_K_M.gguf -ngl 99 \
    --rpc 127.0.0.1:50052 -bs --ctx-checkpoints 0 --cache-ram 0 --host 127.0.0.1 --port 8080
```

llama.cpp lists RPC servers before local GPUs, so when the main host has a GPU it takes the last layers and the output
layer: the logits, sampling and any MTP drafting stay local. If the output layer does end up on `gpu-box` (the main
host's GPU is too small, or has none), `-bs` keeps the logits there.

### Several machines over a VPN

With WireGuard/Tailscale, bind each server to its VPN address only:

```bash
# on 10.8.0.2 and 10.8.0.3
build/bin/ggml-rpc-server --host 10.8.0.2 --port 50052     # use each machine's own VPN address
```

```bash
# main host
GGML_RPC_COMPRESSION=q6 build/bin/llama-cli -m model.gguf -ngl 99 \
    --rpc 10.8.0.2:50052,10.8.0.3:50052
```

Layers are spread across the devices in order; every boundary between two machines is one activation transfer per
ubatch (and per generated token). Between two remote servers the data goes through the main host
(`server A -> main host -> server B`), so that boundary costs two lossy hops.

### Recommended llama-server settings

For any link slower than a LAN, start `llama-server` with:

```bash
--ctx-checkpoints 0 --cache-ram 0
```

The model's KV cache and recurrent state live on the GPUs that hold the layers. Two `llama-server` features copy that
state to the main host, and over RPC every byte crosses the network, from the servers' upload:

- **Context checkpoints** (`--ctx-checkpoints`, default 32 per slot). For models with recurrent layers or
  sliding-window attention (e.g. Qwen3.5/3.6/3.8, Gemma), the server saves the recurrent / SWA state at the end of each
  prompt, so it can later rewind to it. On Qwen3.8-27B one checkpoint is ~150 MB.
- **Host prompt cache** (`--cache-ram`, default 8192 MiB). Before starting a new request, the server copies the whole
  state of the previous one - KV cache and recurrent state - into host RAM, for any model.

Measured on Qwen3.8-27B over RPC, two requests with a 25-token prompt and 128 generated tokens each:

| `llama-server` settings | server -> main host |
|---|---|
| defaults | ~200 MB per request (checkpoint ~134 MB + prompt cache ~71 MB) |
| `--ctx-checkpoints 0` | ~71 MB per request |
| `--ctx-checkpoints 0 --cache-ram 0` | **~100 bytes per generated token** |

On a 50 Mbit/s upload the defaults add ~30 s per request, far more than the generation itself.

What you give up:

- **`--ctx-checkpoints 0`:** a model with recurrent layers cannot rewind its state without a checkpoint. A new request
  still reuses the cached state when its prompt extends the previous conversation exactly, which is the usual
  multi-turn chat case. Editing an earlier message, regenerating a reply or branching a conversation reprocesses the
  whole prompt. Models with only full attention do not create checkpoints, so the flag changes nothing for them.
- **`--cache-ram 0`:** prompts are no longer kept in host RAM when a slot switches to another conversation, so
  returning to an earlier conversation reprocesses it.

Keep the defaults when the servers are on a fast LAN, or when prompt reuse across edited or alternating conversations
matters more than the transfer. The checkpoint save is batched (all its reads share one round trip per server), so on
a fast link it costs only the transfer time.

### Remote output layer: sample on the server (`-bs`)

The model's output layer turns each token's hidden state into logits: one score per vocabulary entry, `n_vocab` f32
values. For Qwen's 151,936-entry vocabulary that is 593.5 KiB per token, 37-150x a hidden state. The output layer goes
with the last device. llama.cpp lists local GPUs after RPC servers, so a main host with a GPU keeps it; when the main
host has no GPU, or its GPU takes no layers, the output layer lands on a remote server, and normally the logits come
back to the main host for every generated token, where the sampler picks the next token. They are never quantized
(sampling and logprobs need exact values) and zstd shrinks them only ~1.2x, so they dominate the traffic: on a
20 Mbit/s server upload they alone cap generation at ~5 tokens/s.

`-bs` / `--backend-sampling` (experimental in llama.cpp) appends the sampler chain to the model's graph as ggml
operations, so it runs on the device that computed the logits. With the whole chain on the server, only the selected
token id - 4 bytes - comes back.

```bash
GGML_RPC_COMPRESSION=q8 build/bin/llama-server -m model.gguf -ngl 99 --rpc 10.8.0.2:50052 -bs \
    --ctx-checkpoints 0 --cache-ram 0
# or: LLAMA_ARG_BACKEND_SAMPLING=1
```

It works with `llama-server`, `llama-cli` and `llama-completion`. Samplers that can run on the device: repetition /
frequency / presence penalties, logit bias, top-k, top-p, min-p, temperature (including dynamic temperature), and the
final `dist` / `greedy` pick. The default sampler settings run entirely on the device: samplers left disabled (DRY,
XTC, typical-p, top-n-sigma at their defaults) become no-ops that are allowed there.

It does not apply, and the logits are read back as usual, when:

- a grammar is active: `--grammar`, JSON schema, or a constrained tool-call format in `llama-server`. The log says
  `backend sampling is not compatible with grammar, disabling`;
- a reasoning budget is set;
- a sampler that only runs on the CPU is enabled (DRY, XTC, typical-p, top-n-sigma, mirostat, adaptive-p): the chain
  runs on the device up to that sampler and the rest on the CPU;
- the application needs the full distribution (perplexity, KL divergence, scoring).

To confirm it is active, run with `GGML_RPC_DEBUG=2 ... -v`. The log shows
`Backend sampler selected token: '...'. Will not run any CPU samplers`, and no `get_tensor_async ... result_output`
reads. Speculative decoding samples its draft model on the device by default (`--spec-draft-backend-sampling`).
See [Backend sampling](#backend-sampling) for measurements.

### Keeping the weight upload cheap

Weights are sent once per load. zstd shrinks them by 1.5-20% depending on the quantization type (IQ quants compress
best). The servers also keep a disk cache of weight tensors of 1 MiB or more, **on by default**: loading the same
model again sends almost nothing.

```bash
build/bin/ggml-rpc-server --host 127.0.0.1 --port 50052 --cache-size 200   # optional limit in GiB
build/bin/ggml-rpc-server --no-cache                                        # disable it
```

Measured with Qwen3-8B Q4_K_M:

| | uploaded to the server | at 50 Mbit/s |
|---|---|---|
| first load | 4,582 MB | ~12 min |
| later loads | **2.5 MB** | < 1 s |

Share of each model's weights held in tensors of 1 MiB or more: 99.98% for Qwen3-8B Q4_K_M, 99.98% for
Qwen3.8-Flash-Next Q8_0 (32 MB of its 188 GB are sent on every load).

How it works:

- **Content-addressed, named by the server.** An entry's name is the digest of its data: SHA-256 over 4 MiB chunks
  (hashed in parallel), then over the chunk digests, the size and a domain tag. The server computes it from the data
  it received, so a client cannot make another client's model load different data.
- **One round trip per load.** On its first cacheable tensor the client fetches the list of entries (`CACHE_LIST`).
  For each listed tensor it then sends `SET_TENSOR_CACHED` without waiting for the reply; the server loads the entry
  from disk. Other tensors are sent normally and stored.
- **Eviction.** Least recently used entries go first, to stay under `--cache-size` (no fixed limit by default) and to
  always keep 10% of the disk free. Entries listed to or used by the connected client are never evicted while it is
  connected; if nothing else can go, new tensors are simply not cached.
- **Housekeeping.** Writes go to a temporary file renamed into place, so an interrupted write never leaves a truncated
  entry. At startup the server removes leftover temporary files and entries of the previous cache format (64-bit FNV
  names), which can never be used again.

**Stored digests: no local reading either.** The main host also remembers the digest of every tensor it uploads, in
a small index per model file under `$LLAMA_CACHE/rpc-digests/` (default `~/.cache/llama.cpp/rpc-digests/`). When the
model is loaded again, the loader looks a tensor up there before reading it; if the server's cache holds that digest,
the tensor is set from the server's cache and **never read from the local disk or hashed**. Only the small tensors and
anything the server no longer holds are read and sent. Measured with Qwen3-8B Q4_K_M, second load:

| | client CPU time | wall time | uploaded |
|---|---|---|---|
| without stored digests (`LLAMA_RPC_DIGESTS=0`) | 4.1 s | 3.7 s | 2.5 MB |
| with stored digests | **0.4 s** | **2.5 s** | 2.5 MB |

For a model that is not in the OS page cache the difference is larger: without stored digests the whole model is read
from disk and hashed (~9 GB/s with the x86 SHA extensions, ~1.7 GB/s without them - ~20 s of hashing alone for a
188 GB model); with them only the ~32 MB of small tensors are read.

An index belongs to one model file, identified by its path, size and modification time; a file that changes in any of
those gets a new index. If the server has evicted an entry, the tensor is simply read and uploaded again. The index
trusts that a file with the same path, size and modification time has the same content - a file rewritten in place
while keeping both (not something downloads, copies or `llama-quantize` do) would load the old weights. Set
`LLAMA_RPC_DIGESTS=0` to always read and hash. Tensors are always read when `--check-tensors` is used.

### Forbidding lossy transfers on a server

```bash
build/bin/ggml-rpc-server -z zstd    # clients asking for q8/q6/q4 get zstd
```

### Checking what is sent

`GGML_RPC_DEBUG=3` logs every compressed payload with its sizes and codec. On the client the RPC debug messages go
through llama.cpp's logger, so also pass `-v`:

```bash
GGML_RPC_DEBUG=3 GGML_RPC_COMPRESSION=q4 build/bin/llama-completion -m Qwen3-0.6B-Q4_K_M.gguf -ngl 99 \
    --rpc 127.0.0.1:50052 -v -p "The capital of France is" -n 8
```

An excerpt for one generated token (Qwen3-0.6B, the whole model on one server):

```
[serialize_set_tensor_compressed] tensor RPC0[127.0.0.1:50052]#embd#0 compressed 4.00 KiB -> 617 B (codec 5)
[serialize_set_tensor_compressed] tensor RPC0[127.0.0.1:50052]#node_1#0 compressed 4.00 KiB -> 49 B (codec 133)
[serialize_set_tensor_compressed] tensor RPC0[127.0.0.1:50052]#attn_inp_kq_mask#0 compressed 512 B -> 33 B (codec 2)
[rpc_unpack_get_response] GET_TENSOR decompressed 487.18 KiB -> 593.50 KiB
```

The input embedding is quantized to q4 (4 KiB -> 617 B), an unused all-zero input collapses to 49 bytes, the mask
goes lossless, and the logits come back lossless - and dominate the traffic (see
[Estimating bandwidth](#estimating-bandwidth)).

Codec ids: 1 zstd, 2 shuffled zstd (floats), 3/4/5 q8/q6/q4; +128 when the lossy stream is additionally zstd-wrapped.

### Environment variables

| Variable | Where | Meaning |
|----------|-------|---------|
| `GGML_RPC_COMPRESSION` | client (and server if `-z` is not given) | most aggressive level to use / accept. Client default `zstd`; server default `q4` (accept all) |
| `GGML_RPC_DEBUG` | both | log verbosity; `2` traces every tensor transfer, `3` adds every compressed payload |
| `LLAMA_ARG_BACKEND_SAMPLING` | client | `1` is the same as `-bs` |
| `GGML_RPC_NO_PIPELINE` | client | set to disable request pipelining (see [Round trips](#round-trips)); for debugging |
| `LLAMA_RPC_DIGESTS` | client | `0` disables the stored weight digests (see [Keeping the weight upload cheap](#keeping-the-weight-upload-cheap)) |
| `LLAMA_CACHE` | both | cache directory: the server's weight cache (`rpc/`) and the client's digest index (`rpc-digests/`) |

## Estimating bandwidth

A layer split across machines moves one hidden-state row of `n_embd` values per token at each machine boundary.
Approximate bytes per token per hop:

| `n_embd` (example models) | f32 raw | zstd | q8 | q6 | q4 |
|---------------------------|---------|------|----|----|----|
| 1024 (Qwen3-0.6B) | 4.0 KiB | 3.4 KiB | 1.0 KiB | 0.8 KiB | 0.6 KiB |
| 4096 (Qwen3-8B, Llama-3-8B) | 16 KiB | 13.4 KiB | 4.0 KiB | 3.1 KiB | 2.3 KiB |
| 5120 (Qwen3-32B) | 20 KiB | 17 KiB | 5.0 KiB | 3.9 KiB | 2.8 KiB |
| 8192 (Llama-3-70B) | 32 KiB | 27 KiB | 7.9 KiB | 6.2 KiB | 4.5 KiB |

Prompt processing moves a whole ubatch at once: with `n_embd = 4096` and the default ubatch of 512 tokens, one hop is
8 MiB raw, ~2.0 MiB at q8 and ~1.15 MiB at q4. On a 100 Mbit/s link that is ~670 ms vs ~170 ms vs ~95 ms per hop per
ubatch.

During generation a hop carries a few KiB, so the round-trip latency between machines usually dominates; compression
pays off most for prompt processing, batched requests and thin links.

### Round trips

Each remote server costs **one network round trip per generated token**: the main host has to read the server's
result before it can continue. With a 30 ms round trip between homes, two servers cap generation at ~16 tokens/s
before any compute, whatever the compression level.

On TCP connections the client pipelines requests: one thread sends each request as soon as it is queued while a
second thread reads the responses in order, so reads issued back to back share a single round trip. This matters
for `-bs`, which reads three small results from the last server per token. Measured with an injected 30 ms round trip
(Qwen3-0.6B, one server, 32 tokens):

| | pipelining off | pipelining on |
|---|---|---|
| without `-bs` | 35.7 ms/token | 35.9 ms/token |
| with `-bs` | 96.0 ms/token (3 round trips) | **34.3 ms/token (1 round trip)** |

Outputs were byte-identical in all configurations. With no added latency the extra thread hand-off costs ~0.1 ms per
token. RDMA connections keep the single-threaded path.

Two things can dwarf the hidden states:

- **Logits.** If the model's output layer runs on a remote server, every generated token pulls back `n_vocab` f32
  logits - ~600 KB for Qwen's 151k vocabulary, versus 4 KiB for a 1024-wide hidden state. Logits stay lossless (zstd
  shrinks them only ~1.19x). Use [`-bs`](#remote-output-layer-sample-on-the-server--bs) so only the sampled token id
  comes back, or keep the output layer on the main host, e.g. by having a local GPU take the last layers.
- **Unused graph inputs.** llama.cpp's input-embedding graph may send inputs that the batch does not use (all zeros).
  They cost almost nothing - the zstd stage compresses each 2 MiB copy to under 100 bytes - but they still take a round
  trip.

## Measurements

All measurements were taken on one machine (RTX 5000 Ada Laptop GPU, 32-thread CPU), with RPC servers on localhost
behind a byte-counting TCP proxy.

### Method

- **Codec accuracy** was measured on real activations: the residual stream (`l_out-*`, the tensor that crosses machine
  boundaries in a layer split) captured from Qwen3-0.6B and Qwen3-8B (Q4_K_M) over a 512-token prompt plus decoded
  tokens. Two errors are reported, averaged per token row: *row error* `||y-x|| / ||x||`, and *bulk error*, the same
  over the values below the 99th percentile of each row - the few massive channels otherwise hide damage to the rest.
- **Model quality** was measured with `llama-perplexity --kl-divergence` against an uncompressed baseline, 20 chunks of
  512 tokens, model split across **two RPC servers** with a CPU-only main host. Each forward pass makes **three lossy
  hops**: input embeddings to server A, the hidden state read from A, and written to B. Logits come back lossless.
- **Exact compute for KLD.** On GPU servers with a quantized model, CUDA re-quantizes activations to q8_1 inside its
  matmuls, so *any* perturbation of the input - even rounding it to fp16 - re-rolls that internal rounding in every
  layer and produces a fixed KLD of 0.003-0.008 regardless of codec. To measure transmission error alone, the precise
  results below use Qwen3-0.6B **dequantized to F32**, servers on CPU, `-ctk f32 -ctv f32 -fa off`, so every operation
  is plain f32 arithmetic. Under these conditions, sending every activation as fp16 gives KLD < 1e-6.

### Codec accuracy and size on real activations

| Level | Model | Row error | Bulk error | Ratio vs f32 | Encode | Decode |
|-------|-------|-----------|------------|--------------|--------|--------|
| q8 | Qwen3-0.6B | 0.0049 | 0.0064 | 3.96x | 510 MB/s | 3.1 GB/s |
| q6 | Qwen3-0.6B | 0.0197 | 0.0257 | 5.00x | 468 MB/s | 3.8 GB/s |
| q4 | Qwen3-0.6B | 0.0566 | 0.0729 | 6.94x | 232 MB/s | 4.2 GB/s |
| q8 | Qwen3-8B | 0.0052 | 0.0077 | 4.07x | 746 MB/s | 3.8 GB/s |
| q6 | Qwen3-8B | 0.0208 | 0.0306 | 5.14x | 690 MB/s | 3.7 GB/s |
| q4 | Qwen3-8B | 0.0535 | 0.0783 | 7.12x | 399 MB/s | 5.1 GB/s |

Throughput is per payload with up to 8 threads (rows are split across threads for payloads over 128k values; a single
generated token is encoded on one thread in microseconds).

### Model quality (Qwen3-0.6B F32, exact compute)

| Level | Mean KLD | Same top token | PPL ratio |
|-------|----------|----------------|-----------|
| zstd | 0 (bit-exact) | 100% | 1.0000 |
| q8 | 0.000039 | 99.61% | +0.002% |
| q6 | 0.000597 | 98.77% | +0.05% |
| q4 | 0.006281 | 96.35% | +0.06% |
| *previous fp8 codec* | 0.008224 | 95.98% | +0.81% |
| *previous fp4 codec* | 0.921555 | 64.88% | 2.77x (model broken) |

PPL ratios are relative to the bit-exact `zstd` run. The previous fp8/fp4 rows were obtained by quantizing the same activations with the old algorithm (one scale per
payload, fp8 with a rounding bug) and sending them losslessly, which reproduces its accuracy exactly. KLD scales with
the square of the codec error, as expected: q6/q8 is 15x (bulk error ratio squared: 16x).

### Model quality on GPU (Qwen3-8B Q4_K_M, CUDA servers)

The realistic configuration. KLD here includes the GPU rounding floor described above, so it overstates the cost of q8
and cannot separate small differences.

| Level | Mean KLD | Same top token |
|-------|----------|----------------|
| zstd | 0 (bit-exact) | 100% |
| q8 | 0.0025 | 97.9% |
| q4 | 0.0047 | 97.6% |

### Bytes on the wire

Activation traffic main host -> servers during evaluation (10,240 tokens, two servers; includes masks, positions and
llama.cpp's unused zero inputs, excludes the weight upload):

| Level | Qwen3-0.6B F32 | vs none | Qwen3-8B Q4_K_M | vs none |
|-------|----------------|---------|-----------------|---------|
| none | 372.1 MB | 1x | 692.9 MB | 1x |
| zstd | 66.6 MB | 5.6x | 217.7 MB | 3.2x |
| q8 | 32.6 MB | 11.4x | 74.3 MB | 9.3x |
| q6 | 29.0 MB | 12.8x | - | - |
| q4 | 25.4 MB | 14.6x | 44.1 MB | 15.7x |

The reduction exceeds the per-tensor ratio because the zero inputs and masks almost vanish. The download direction in
this benchmark (~2.6 GB) is almost entirely logits: `llama-perplexity` reads the logits of every token, and the output
layer sat on a remote server.

Per-tensor sizes for one 512-token ubatch of Qwen3-0.6B (2 MiB of f32 hidden state):

| Tensor | zstd | q8 | q4 |
|--------|------|----|----|
| hidden state (`l_out-14`) | 1.69 MiB | 538 KiB | 298 KiB |
| attention mask (1 MiB, f32 with `-inf`) | 2.2 KiB | 2.2 KiB (lossless) | 2.2 KiB (lossless) |
| unused zero input (2 MiB) | 93 B | < 1 KiB | < 1 KiB |

### Backend sampling

Qwen3-0.6B Q4_K_M, the whole model on one RPC server, CPU-only main host, `llama-completion` generating 64 tokens at
`q8` with the default sampler chain:

| | without `-bs` | with `-bs` |
|---|---|---|
| logits reads from the server | 66 (593.5 KiB each) | 0 |
| server -> main host bytes, whole run | 29.4 MB | 3.2 KB |
| generation speed (localhost) | 4.90 ms/token | 2.68 ms/token (1.8x faster) |
| generation speed, 30 ms round trip | 35.9 ms/token | 34.3 ms/token (with request pipelining, see [Round trips](#round-trips)) |
| greedy output (`--temp 0`) | - | identical |
| random sampling, same seed | - | identical in this run |

The speedup on localhost, where bandwidth is no constraint, comes from no longer compressing ~600 KB of logits on the
server and decompressing them on the main host for every token; on a real link the gain is larger. With `--grammar`,
backend sampling disabled itself as expected and the logits were read back.

### Weights (lossless, sent once)

zstd level 3 on 256 MiB samples of model files:

| Model file | Ratio |
|------------|-------|
| Qwen3-8B-Q4_K_M | 1.038x |
| Qwen3.6-35B-A3B-UD-IQ3_S | 1.203x |

### Lossless codec choice for floats

zstd on raw f32 activations gains only ~8%: the mantissa bytes are noise. Shuffling the bytes into planes (all byte 0s,
then all byte 1s, ...) lets zstd exploit the sign/exponent bytes:

| | zstd -1 | zstd -3 |
|---|---|---|
| plain f32 activations | 1.080x, 1.0-1.2 GB/s | 1.080x |
| byte-shuffled | **1.19x**, ~1 GB/s | 1.17x, 0.5-0.6 GB/s |

Level 1 on shuffled data is both faster and smaller than level 3, so float payloads use it.

### Design comparison (why these codecs)

Candidate codecs evaluated on the same real activations (bulk error, lower is better; bits per value include all
scales and outliers):

| Codec | Bits/value | Bulk error 0.6B | Bulk error 8B |
|-------|------------|-----------------|---------------|
| fp4 (e2m1), one scale per row | 4.0 | 0.414 | 0.781 |
| NVFP4-style: e2m1, blocks of 16, e4m3 scale + search | 4.5 | 0.110 | 0.118 |
| int4, blocks of 16 | 4.5 | 0.137 | 0.149 |
| IQ4_NL grid, blocks of 16 | 4.5 | 0.117 | 0.133 |
| IQ4_NL + Hadamard rotation | 4.5 | 0.115 | 0.124 |
| IQ4_NL + exact outliers | 4.69 | 0.078 | 0.084 |
| **IQ4_NL + exact outliers + signed scale (q4)** | 4.63 | **0.074** | **0.079** |
| e4m3 elements, one scale per row | 8.0 | 0.026 | 0.027 |
| int8, one scale per row | 8.0 | 0.048 | 0.116 |
| int8, blocks of 32, e4m3 scale | 8.25 | 0.014 | 0.016 |
| **int8, blocks of 32 + exact outliers (q8)** | 8.38 | **0.0064** | **0.0077** |

Findings that shaped the design: a single scale per row or payload fails on real activations; fine block scales fix
most of it; removing the handful of massive channels before quantizing halves the remaining error; with outliers
removed, the non-uniform IQ4_NL grid beats both fp4 and int4; Hadamard rotation lowers total error but spreads the
outliers' error over the bulk, so it was not used. A least-squares refit of the block scale gave no measurable
improvement and was dropped. Narrowing the q4 scale search from +/-2 to +/-1 steps encodes 30% faster but raises the
bulk error by 1.4%; +/-2 was kept, favouring accuracy.

## How it works

LLM hidden states have a few "massive" channels - for Qwen3 up to ~14,000 while typical values are ~1 - and they sit at
fixed positions. Spending one scale on a whole row wastes nearly all quantization levels on them. Each lossy level
therefore splits every row into three parts:

1. **Exact outliers.** The largest `max(4, n/256)` values of the row by magnitude are sent exactly (16-bit index plus an
   f16 value, or f32 if any value exceeds the f16 range) and zeroed before quantization.
2. **Row scale.** An f32 scale `g` per row, set so the largest remaining value maps to the top code.
3. **Blocks.** Every block of 32 (q8, q6) or 16 (q4) values gets its own 8-bit e4m3 scale relative to `g`:
   - q8 / q6: symmetric integers in [-127, 127] / [-31, 31]. The block scale is the smallest e4m3 value that does not
     clip (q6 also tries one step lower and keeps whichever has less error).
   - q4: indices into the IQ4_NL grid `{-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}`. The
     grid is asymmetric, so the scale is signed; the encoder tries 5 scale values of each sign around the nearest and
     keeps the one with the least squared error. The nearest grid index comes from a 512-entry table (the midpoints
     between grid values fall on multiples of 0.5, so the table is exact).

The encoded stream is then compressed with zstd level 1 and kept in that form when smaller: it saves 0-6% on real
activations and collapses constant or zero payloads to almost nothing.

Payloads that contain a non-finite value (attention masks use `-inf`) take the lossless path instead: there is no code
for them, and masks compress far better losslessly.

The decoder reconstructs the values in the tensor's own type (f32, f16 or bf16).

## Wire format

Intended for maintainers; protocol version stays 8.1.0, the compression codec set has its own version.

**Negotiation.** The last byte of the HELLO `conn_caps` carries `(codec_set_version << 4) | level`, codec set version
2. A peer advertising a different version (including all-zero caps from builds without compression, or the earlier
fp8/fp4 build) negotiates `none`. Both sides use `min(local, remote)`. Builds with the RDMA transport compiled in use
the whole caps array for RDMA and never compress.

**Features.** The byte before it carries feature flags; bit 0 (`RPC_FEAT_CACHE`) means the server has a disk cache and
serves `CACHE_LIST` and `SET_TENSOR_CACHED`. Peers without it send zero, and the client then does not use the cache.

**SET_TENSOR**: `| rpc_tensor | flags (u8) | offset (u64) | data |`. Flag bit 0: store in the server cache (the server
names the entry from its own digest of the data); bit 1: data is a compressed payload.

**CACHE_LIST**: empty request; the response is the 32-byte digests of all cached entries, which stay protected from
eviction for the connection. **SET_TENSOR_CACHED**: `| rpc_tensor | offset (u64) | size (u64) | digest (32 bytes) |`,
response one status byte. The client sends it without waiting and the dispatcher aborts if the status is zero.
**SET_TENSOR_HASH** (64-bit FNV, previous protocol) always answers "miss": a client could collide such a hash on
purpose. **SET_TENSOR_2D**: when compression was negotiated, a flags byte follows the 4 x u64 header.

**GET_TENSOR / GET_TENSOR_2D**: when compression was negotiated, the request carries an extra `u64 level` - the most
aggressive level the client allows for that payload (the server cannot tell activations from weights or outputs) - and
the response is `| flags (u8) | data |`, bit 1 meaning compressed.

**Compressed payload**: `| codec (u8) | uncompressed size (u64) | body |`

| Codec | Body |
|-------|------|
| 0 raw | the bytes |
| 1 zstd | one zstd frame |
| 2 shuffled zstd | `| element size (u8) | zstd frame of the byte-shuffled data |` |
| 3 / 4 / 5 q8 / q6 / q4 | lossy stream (below) |
| 131 / 132 / 133 | lossy stream wrapped in one zstd frame (codec + 128) |

**Lossy stream**: `| row_len (u32) | n_rows (u32) | k (u16) | flags (u8, bit 0: f32 outliers) | reserved (u8) |` then
`n_rows` rows of identical size:
`| g (f32) | outlier index (u16 x k) | outlier value (f16 or f32 x k) | block scale (e4m3 x nb) | codes |`,
codes bit-packed LSB first (8, 6 or 4 bits each).

**Validation.** The receiver always knows the uncompressed size and type, and rejects any payload whose declared size,
row geometry, outlier indices or zstd frame size disagree with it. The server checks the write region against the
destination buffer before allocating anything for decompression. The codec tests fuzz all codecs with truncated and
bit-flipped payloads under AddressSanitizer and UBSan.

## Limitations and known issues

- **No security.** No authentication or encryption; use a tunnel or VPN. The weight cache is content-addressed with a
  cryptographic digest computed by the server, so clients cannot poison each other's entries, but any client can fill
  the cache with data of its own (bounded by `--cache-size` and the 10% free-disk rule).
- **Both sides must run this build** to use compression. Older peers interoperate without compression.
- **Logits from a remote output layer** are large and lossless. `-bs` avoids them for generation (see
  [Remote output layer](#remote-output-layer-sample-on-the-server--bs)) but is experimental in llama.cpp and does not
  apply with grammars, a reasoning budget or CPU-only samplers. With models whose output weight is tied to the token
  embeddings (e.g. Qwen3-0.6B), `-ot` overrides did not move the output computation to the main host in testing.
  Backend sampling was tested with `llama-completion` and one server, not with `llama-server` or several servers.
- **`llama-server` state copies.** Context checkpoints and the host prompt cache copy the KV cache and recurrent state
  from the servers to the main host, up to ~200 MB per request; use `--ctx-checkpoints 0 --cache-ram 0` on slow links
  (see [Recommended llama-server settings](#recommended-llama-server-settings)).
- **RDMA builds** do not compress, even when the connection falls back to TCP.
- **Server-to-server boundaries** go through the main host, so they cost two lossy hops.
- **CPU cost.** q4 encodes at 230-400 MB/s and q8 at 500-750 MB/s. On a fast LAN (multi-gigabit) the encode time can
  exceed the transfer time it saves; use `zstd` or `none` there.
- **Measuring quality on GPU** with quantized models hits a noise floor (see [Method](#method)); compare levels on an
  F32 model with CPU servers.

## Tests

```bash
cmake -B build -DGGML_RPC=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build -j --target test-rpc-compress-codec test-rpc-cache test-rpc-compression ggml-rpc-server

# codec unit tests: lossless round trips, per-level error and ratio bounds,
# outliers, masks, zero payloads, corrupt payloads
build/bin/test-rpc-compress-codec

# weight cache: SHA-256 test vectors, cache digest, disk cache store / load /
# eviction / protection / cleanup
build/bin/test-rpc-cache

# end to end against real servers: negotiation across 10 server/client level
# combinations; weights, KV cache, outputs and masks bit-exact; activations
# quantized within each level's bound
bash tests/test-rpc-compression.sh build/bin/ggml-rpc-server build/bin/test-rpc-compression
```

Reproducing the model-quality numbers:

```bash
# F32 copy of the model, so the compute itself adds no rounding
build/bin/llama-quantize --allow-requantize Qwen3-0.6B-Q4_K_M.gguf qwen3-0.6b-f32.gguf F32

build/bin/ggml-rpc-server --device CPU --port 50061 &
build/bin/ggml-rpc-server --device CPU --port 50062 &

ARGS="-m qwen3-0.6b-f32.gguf -f wiki.test.raw -c 512 --chunks 20 -ngl 99 --rpc 127.0.0.1:50061,127.0.0.1:50062 -ctk f32 -ctv f32 -fa off"
GGML_RPC_COMPRESSION=none CUDA_VISIBLE_DEVICES= build/bin/llama-perplexity $ARGS --kl-divergence-base base.kld
for l in zstd q8 q6 q4; do
    GGML_RPC_COMPRESSION=$l CUDA_VISIBLE_DEVICES= build/bin/llama-perplexity $ARGS --kl-divergence-base base.kld --kl-divergence
done
```
