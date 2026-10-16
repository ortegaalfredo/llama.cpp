// Compression for the RPC transport (TCP only).
//
// Only the bulk data paths are compressed: the tensor payloads of
// SET_TENSOR / SET_TENSOR_2D / GET_TENSOR / GET_TENSOR_2D.  Small control
// messages (graph descriptions, alloc requests, ...) stay uncompressed.
//
// Levels, ordered by increasing aggressiveness. Both peers advertise the most
// aggressive level they accept and the connection uses the lower of the two:
//
//   none - raw passthrough
//   zstd - lossless. Float payloads are byte-shuffled first (sign/exponent
//          bytes compress much better on their own plane), everything else
//          (quantized weights, integer inputs) goes to zstd as is.
//   q8   - lossy, ~8.4 bits/value: int8 in blocks of 32
//   q6   - lossy, ~6.4 bits/value: int6 in blocks of 32
//   q4   - lossy, ~4.6 bits/value: the IQ4_NL non-linear grid in blocks of 16
//
// The lossy levels only apply to activations - see the caller for the exact
// policy. Weights, the KV cache, graph outputs (logits, embeddings) and any
// payload with a non-finite value always take the lossless path.
//
// Lossy layout. LLM activations have a handful of "massive" channels that
// are 100-1000x larger than the rest, and they sit at fixed positions. A
// single scale per row or per payload wastes nearly all quantization levels
// on them, so the lossy codecs use three pieces per row:
//
//   - the top max(4, row_len/256) values by magnitude are sent exactly (index +
//     f16/f32 value) and removed from the row before quantization
//   - a per-row f32 scale g
//   - a per-block e4m3 scale relative to g, so every block of 16/32 values
//     gets its own range
//
// Per row: | g (f32) | idx (u16 x k) | val (f16|f32 x k) | block scales (u8 x nb) | codes (bit-packed) |

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ggml.h"

// compression levels, ordered by increasing aggressiveness
enum rpc_compress_level : uint8_t {
    RPC_COMP_NONE = 0,
    RPC_COMP_ZSTD = 1,
    RPC_COMP_Q8   = 2,
    RPC_COMP_Q6   = 3,
    RPC_COMP_Q4   = 4,
    RPC_COMP_COUNT,
};

// codec id stored in the first byte of a compressed payload
enum rpc_codec : uint8_t {
    RPC_CODEC_RAW       = 0,   // stored verbatim
    RPC_CODEC_ZSTD      = 1,   // zstd frame
    RPC_CODEC_ZSTD_SHUF = 2,   // | elem size (u8) | zstd frame of the byte-shuffled data |
    RPC_CODEC_Q8        = 3,
    RPC_CODEC_Q6        = 4,
    RPC_CODEC_Q4        = 5,
};

// set on a lossy codec id when its stream is wrapped in a zstd frame: the
// codes of real activations are not uniformly distributed (a few percent),
// and payloads full of zeros or constants collapse to almost nothing
static constexpr uint8_t RPC_CODEC_FLAG_ZSTD = 0x80;

// wire header of a compressed payload: | codec (1 byte) | orig_size (8 bytes LE) | payload |
static constexpr size_t RPC_COMP_HEADER_SIZE = 9;

// the lossy codecs only take rows at least this wide; narrower float tensors
// are per-token scalars (attention scales, pooling weights, ...) and are sent
// losslessly
static constexpr uint32_t RPC_COMP_MIN_ROW = 256;

// The level is advertised in one byte of the HELLO conn_caps: the high nibble
// is the version of the codec set, the low nibble the level. A peer with a
// different codec set (or none) negotiates RPC_COMP_NONE.
static constexpr uint8_t RPC_COMP_CAPS_VERSION = 2;

static inline bool rpc_compress_level_valid(uint8_t level) {
    return level < RPC_COMP_COUNT;
}

static inline bool rpc_compress_level_lossy(uint8_t level) {
    return level >= RPC_COMP_Q8 && level < RPC_COMP_COUNT;
}

const char * rpc_compress_level_name(uint8_t level);

// parses none / zstd / q8 / q6 / q4 (fp8 and fp4 are accepted as aliases of
// q8 and q4) or a number. Returns false for anything else.
bool rpc_compress_level_parse(const char * str, uint8_t & level);

// the compression level this process asks for, from GGML_RPC_COMPRESSION.
// Defaults to zstd: lossless, so it is always safe.
uint8_t rpc_compress_env_level();

// packs a payload of `size` bytes of ggml type `type` for transmission.
// `level` is the most aggressive level allowed for this payload; a lossy
// level is only used for float types whose size is a whole number of rows
// of `row_len` (>= RPC_COMP_MIN_ROW) elements, and only if every value is
// finite - otherwise the payload falls back to the lossless codec.
// On success `out` holds the wire bytes (header + payload) and the return
// value is the codec used. Returns RPC_CODEC_RAW to mean "send the data as
// is", in which case `out` is left untouched.
uint8_t rpc_compress_payload(std::vector<uint8_t> & out, const void * data, size_t size,
                             enum ggml_type type, uint32_t row_len, uint8_t level);

// unpacks a compressed payload into `out` (out_size bytes). The receiver
// always knows the uncompressed size, so a corrupt or hostile payload cannot
// make this allocate or write an arbitrary amount. `type` is the tensor type
// the payload belongs to; a lossy payload is reconstructed in that type.
// Returns false on any mismatch.
bool rpc_decompress_payload(void * out, size_t out_size, const void * data, size_t size,
                            enum ggml_type type);

// reads the uncompressed size from a compressed payload's header
bool rpc_compressed_orig_size(const void * data, size_t size, uint64_t & orig_size);
