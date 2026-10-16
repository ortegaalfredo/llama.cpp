#!/usr/bin/env bash
# Tests RPC transport compression end to end: for each case starts a real rpc
# server, connects a client with its own level, checks the level both sides
# negotiate and the payload policy at that level (see test-rpc-compression.cpp).
set -uo pipefail

server=$1
client=$2
port=$((42000 + $$ % 3000))
endpoint="127.0.0.1:${port}"
test_dir=$(mktemp -d)
rc=0

cleanup() {
    kill "${pid:-}" 2>/dev/null || true
    rm -rf "$test_dir"
}
trap cleanup EXIT

wait_for_port() {
    for _ in {1..200}; do
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            exec 3>&- 3<&-
            return 0
        fi
        sleep 0.05
    done
    return 1
}

# server level / client level / level the connection must negotiate
# ("default" starts the server without --compression: it accepts any level)
cases=(
    "none none none"
    "zstd zstd zstd"
    "q8 q8 q8"
    "q6 q6 q6"
    "q4 q4 q4"
    "default q6 q6"
    "q8 q4 q8"
    "none q4 none"
    "q4 zstd zstd"
    "fp8 fp4 q8"
)
for c in "${cases[@]}"; do
    read -r srv_level cli_level expected <<< "$c"
    echo "=== server: $srv_level, client: $cli_level, expected: $expected ==="
    srv_args=()
    if [ "$srv_level" != "default" ]; then
        srv_args=(--compression "$srv_level")
    fi
    log="$test_dir/server-$srv_level-$cli_level.log"
    env -u GGML_RPC_COMPRESSION "$server" --device CPU --host 127.0.0.1 --port "$port" "${srv_args[@]}" >"$log" 2>&1 &
    pid=$!
    if ! wait_for_port; then
        echo "server did not start"
        cat "$log"
        rc=1
        break
    fi
    # the client picks its level up from the environment
    GGML_RPC_COMPRESSION="$cli_level" "$client" "$endpoint" "$expected"
    client_rc=$?
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    if [ $client_rc -ne 0 ]; then
        echo "client failed"
        cat "$log"
        rc=1
        break
    fi
    if ! grep -q "compression: $expected" "$log"; then
        echo "server did not negotiate $expected:"
        cat "$log"
        rc=1
        break
    fi
    echo "--- ok"
done

if [ $rc -eq 0 ]; then
    echo "ALL COMPRESSION CASES PASSED"
fi
exit $rc
