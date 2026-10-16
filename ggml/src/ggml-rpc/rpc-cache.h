// Server-side weight cache for the RPC backend.
//
// Weight tensors are cached on the server's disk under the name of their
// content digest, so a model loaded again does not travel over the network.
//
// Digest: SHA-256 over 4 MiB chunks, hashed in parallel, then
//   SHA-256("ggml-rpc-cache-v1" || size (u64 LE) || chunk digests...)
// It is collision resistant, so a client cannot craft data that takes the
// place of another model's tensor, and the server always names entries from
// its own digest of the data it received.
//
// The cache keeps its size under an optional limit and always leaves 10% of
// the disk free, evicting the least recently used entries. Entries a client
// has been told about (list) or has used in the current connection are never
// evicted during that connection.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct rpc_digest {
    std::array<uint8_t, 32> b {};

    bool operator==(const rpc_digest & other) const { return b == other.b; }
    std::string hex() const;
    static bool from_hex(const std::string & s, rpc_digest & out);
};

// the digest is already uniformly distributed: its first bytes make a good hash
struct rpc_digest_hash {
    size_t operator()(const rpc_digest & d) const {
        size_t h;
        memcpy(&h, d.b.data(), sizeof(h));
        return h;
    }
};

// plain SHA-256, exposed for tests
void rpc_sha256(const void * data, size_t size, uint8_t out[32]);

// the cache digest of a tensor's data (see above)
rpc_digest rpc_cache_digest(const void * data, size_t size);

class rpc_disk_cache {
public:
    // max_bytes = 0: no fixed limit (the 10% free disk rule still applies)
    rpc_disk_cache(const std::string & dir, uint64_t max_bytes);

    // starts a client connection: entries protected for the previous
    // connection become evictable again
    void begin_session();

    // digests of all cached entries; they stay protected for this connection
    std::vector<rpc_digest> list();

    // reads an entry of exactly `size` bytes; false if it is missing or has
    // another size
    bool load(const rpc_digest & digest, uint64_t size, std::vector<uint8_t> & out);

    // stores data under its digest unless already present; false if there is
    // no room for it
    bool store(const void * data, size_t size);

    uint64_t total_size() const { return total; }
    size_t   n_entries()  const { return index.size(); }

private:
    struct entry {
        uint64_t size;
        int64_t  last_use; // file clock ticks
    };

    std::string path_of(const rpc_digest & digest) const;
    void touch(const rpc_digest & digest);
    bool ensure_space(uint64_t need);

    std::string dir;
    uint64_t    max_bytes;
    uint64_t    total = 0;
    std::unordered_map<rpc_digest, entry, rpc_digest_hash> index;
    std::unordered_set<rpc_digest, rpc_digest_hash>        protected_entries;
    bool        warned_full = false;
    std::mutex  mutex;
};
