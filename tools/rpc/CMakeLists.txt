set(TARGET ggml-rpc-server)
add_executable(${TARGET} rpc-server.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_BUILD_TESTS AND UNIX AND NOT GGML_BACKEND_DL)
    add_executable(test-rpc-multi-server ${PROJECT_SOURCE_DIR}/tests/test-rpc-multi-server.cpp)
    target_link_libraries(test-rpc-multi-server PRIVATE ggml ggml-rpc)
    target_include_directories(test-rpc-multi-server PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    add_test(
        NAME test-rpc-multi-server
        COMMAND bash ${PROJECT_SOURCE_DIR}/tests/test-rpc-multi-server.sh
            $<TARGET_FILE:ggml-rpc-server>
            $<TARGET_FILE:test-rpc-multi-server>)
    set_property(TEST test-rpc-multi-server PROPERTY LABELS main)

    # transport compression codecs: lossless round trips, lossy accuracy on
    # activation-like data, fallbacks and corrupt payloads
    if (ZSTD_INCLUDE_DIR AND ZSTD_LIB)
        add_executable(test-rpc-compress-codec
            ${PROJECT_SOURCE_DIR}/tests/test-rpc-compress-codec.cpp
            ${PROJECT_SOURCE_DIR}/ggml/src/ggml-rpc/rpc-compress.cpp)
        target_link_libraries(test-rpc-compress-codec PRIVATE ggml ${ZSTD_LIB})
        target_include_directories(test-rpc-compress-codec PRIVATE
            ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-rpc ${ZSTD_INCLUDE_DIR})
        target_compile_definitions(test-rpc-compress-codec PRIVATE GGML_RPC_HAVE_ZSTD)
        add_test(NAME test-rpc-compress-codec COMMAND $<TARGET_FILE:test-rpc-compress-codec>)
        set_property(TEST test-rpc-compress-codec PROPERTY LABELS main)
    endif()

    # weight cache: SHA-256 test vectors, cache digest, disk cache eviction rules
    add_executable(test-rpc-cache
        ${PROJECT_SOURCE_DIR}/tests/test-rpc-cache.cpp
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-rpc/rpc-cache.cpp)
    target_link_libraries(test-rpc-cache PRIVATE ggml)
    target_include_directories(test-rpc-cache PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-rpc)
    target_compile_features(test-rpc-cache PRIVATE cxx_std_17)
    add_test(NAME test-rpc-cache COMMAND $<TARGET_FILE:test-rpc-cache>)
    set_property(TEST test-rpc-cache PROPERTY LABELS main)

    # transport compression end to end: every level against a real server;
    # activations within their level's error, weights and outputs exact
    add_executable(test-rpc-compression ${PROJECT_SOURCE_DIR}/tests/test-rpc-compression.cpp)
    target_link_libraries(test-rpc-compression PRIVATE ggml ggml-rpc)
    target_include_directories(test-rpc-compression PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    add_test(
        NAME test-rpc-compression
        COMMAND bash ${PROJECT_SOURCE_DIR}/tests/test-rpc-compression.sh
            $<TARGET_FILE:ggml-rpc-server>
            $<TARGET_FILE:test-rpc-compression>)
    set_property(TEST test-rpc-compression PROPERTY LABELS main)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
