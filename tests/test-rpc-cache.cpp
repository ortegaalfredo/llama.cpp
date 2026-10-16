// Unit tests for the RPC weight cache (ggml/src/ggml-rpc/rpc-cache.cpp):
// SHA-256 against the FIPS 180-2 test vectors, the chunked cache digest, and
// the disk cache's store / load / eviction / protection rules.
#include "ggml.h"
#include "ggml-rpc/rpc-cache.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static int n_failed = 0;

#define CHECK(cond, ...) do {                         \
    if (!(cond)) {                                    \
        printf("FAIL %s:%d: ", __FILE__, __LINE__);   \
        printf(__VA_ARGS__);                          \
        printf("\n");                                 \
        n_failed++;                                   \
    }                                                 \
} while (0)

static std::string sha256_hex(const void * data, size_t size) {
    rpc_digest d;
    rpc_sha256(data, size, d.b.data());
    return d.hex();
}

static void test_sha256() {
    CHECK(sha256_hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "sha256 empty");
    CHECK(sha256_hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256 abc");
    const char * m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    CHECK(sha256_hex(m, strlen(m)) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", "sha256 448-bit message");
    std::vector<char> a(1000000, 'a');
    CHECK(sha256_hex(a.data(), a.size()) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "sha256 million a");
}

static std::vector<uint8_t> random_bytes(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> v(n);
    for (auto & b : v) {
        b = (uint8_t) rng();
    }
    return v;
}

static void test_digest() {
    // several chunks (4 MiB each), so the parallel path runs
    auto data = random_bytes(10*1024*1024 + 123, 1);
    const rpc_digest d0 = rpc_cache_digest(data.data(), data.size());
    CHECK(d0 == rpc_cache_digest(data.data(), data.size()), "digest is deterministic");

    auto flipped = data;
    flipped[7*1024*1024] ^= 1;
    CHECK(!(d0 == rpc_cache_digest(flipped.data(), flipped.size())), "a flipped bit in a later chunk changes the digest");

    // the size is part of the digest: trailing zeros make a different entry
    auto padded = data;
    padded.push_back(0);
    CHECK(!(d0 == rpc_cache_digest(padded.data(), padded.size())), "the size changes the digest");

    // not a plain SHA-256 of the data
    CHECK(rpc_cache_digest(data.data(), 1024).hex() != sha256_hex(data.data(), 1024), "domain separated from plain sha256");

    rpc_digest parsed;
    CHECK(rpc_digest::from_hex(d0.hex(), parsed) && parsed == d0, "hex round trip");
    CHECK(!rpc_digest::from_hex("xyz", parsed), "rejects bad hex");
    CHECK(!rpc_digest::from_hex(std::string(64, 'G'), parsed), "rejects non-hex characters");
}

static void test_disk_cache(const fs::path & dir) {
    fs::remove_all(dir);
    fs::create_directories(dir);

    const size_t MB = 1024*1024;
    auto a = random_bytes(MB, 10), b = random_bytes(MB, 11), c = random_bytes(MB, 12), d = random_bytes(MB, 13);
    const rpc_digest da = rpc_cache_digest(a.data(), a.size());
    const rpc_digest db = rpc_cache_digest(b.data(), b.size());
    const rpc_digest dc = rpc_cache_digest(c.data(), c.size());
    const rpc_digest dd = rpc_cache_digest(d.data(), d.size());

    // stale files from the old FNV format and interrupted writes are removed at startup
    std::ofstream(dir / "0123456789abcdef") << "old";
    std::ofstream(dir / (std::string(64, 'a') + ".tmp")) << "partial";
    std::ofstream(dir / "unrelated.txt") << "keep";

    {
        // limit of 2.5 entries
        rpc_disk_cache cache(dir.string(), 5*MB/2);
        CHECK(!fs::exists(dir / "0123456789abcdef"), "old FNV entry removed");
        CHECK(!fs::exists(dir / (std::string(64, 'a') + ".tmp")), "temp file removed");
        CHECK(fs::exists(dir / "unrelated.txt"), "unrelated file kept");

        cache.begin_session();
        CHECK(cache.store(a.data(), a.size()), "store a");
        CHECK(cache.store(a.data(), a.size()) && cache.n_entries() == 1, "storing the same data twice keeps one entry");
        CHECK(cache.store(b.data(), b.size()), "store b");
        // a and b are protected in this session: no room for c
        CHECK(!cache.store(c.data(), c.size()), "store c refused while a and b are in use");
        CHECK(cache.n_entries() == 2, "two entries");

        std::vector<uint8_t> out;
        CHECK(cache.load(da, MB, out) && out == a, "load a");
        CHECK(!cache.load(da, MB + 1, out), "load with the wrong size fails");
        CHECK(!cache.load(dc, MB, out), "load of a missing entry fails");

        // next connection: nothing protected. Use b so a is the least recently used
        cache.begin_session();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        CHECK(cache.load(db, MB, out), "load b");
        CHECK(cache.store(c.data(), c.size()), "store c evicts");
        CHECK(!cache.load(da, MB, out), "a (least recently used) was evicted");
        CHECK(cache.load(db, MB, out) && cache.load(dc, MB, out), "b and c remain");
        CHECK(cache.total_size() == 2*MB, "total size tracks evictions");

        // entries listed to a client stay until its connection ends
        cache.begin_session();
        const auto listed = cache.list();
        CHECK(listed.size() == 2, "list returns the entries");
        CHECK(!cache.store(d.data(), d.size()), "listed entries are not evicted");
    }
    {
        // a new process sees the same entries
        rpc_disk_cache cache(dir.string(), 0);
        cache.begin_session();
        std::vector<uint8_t> out;
        CHECK(cache.n_entries() == 2 && cache.load(dc, MB, out) && out == c, "entries persist across restarts");
        CHECK(cache.store(d.data(), d.size()) && cache.load(dd, MB, out), "no fixed limit");
        // the file name is the digest of its content
        std::ifstream f(dir / dd.hex(), std::ios::binary);
        std::vector<uint8_t> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        CHECK(rpc_cache_digest(raw.data(), raw.size()) == dd, "entry content matches its name");
    }
    fs::remove_all(dir);
}

int main(int argc, char ** argv) {
    const fs::path dir = argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path() / "test-rpc-cache";
    printf("sha256:\n");
    test_sha256();
    printf("digest:\n");
    test_digest();
    printf("disk cache:\n");
    test_disk_cache(dir);
    printf("%s\n", n_failed ? "FAILED" : "ALL PASSED");
    return n_failed ? 1 : 0;
}
