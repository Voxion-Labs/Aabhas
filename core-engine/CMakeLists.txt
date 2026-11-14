cmake_minimum_required(VERSION 3.16)

project(aabhas_data_plane LANGUAGES CXX)

option(AABHAS_BUILD_TESTS "Build native data-plane tests" OFF)

add_library(aabhas_data_plane STATIC
    src/memory/madvise_utils.cpp
    src/memory/mmap_illusion.cpp
    src/model/model_index.cpp
    src/tokenizer/tokenizer.cpp
    src/inference/inference_runtime.cpp
    src/inference/kv_cache.cpp
    src/inference/transformer_layers.cpp
    src/inference/quantized_kernels.cpp
    src/scheduler/layer_scheduler.cpp
    src/hardware/nvme_direct.cpp
    src/hardware/nvme_streamer.cpp
    src/hardware/hw_identifier.cpp
    src/engine/native_engine.cpp
    src/tensor_ops/layer_streamer.cpp
)

target_compile_features(aabhas_data_plane PUBLIC cxx_std_17)
target_include_directories(aabhas_data_plane
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(aabhas_data_plane PROPERTIES
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(aabhas_data_plane PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    )
endif()

if (AABHAS_BUILD_TESTS)
    include(CTest)
    find_package(Threads REQUIRED)
    add_executable(aabhas_model_index_test tests/model_index_test.cpp)
    add_executable(aabhas_layer_scheduler_test tests/layer_scheduler_test.cpp)
    add_executable(aabhas_inference_runtime_test tests/inference_runtime_test.cpp)
    add_executable(aabhas_tokenizer_test tests/tokenizer_test.cpp)
    add_executable(aabhas_transformer_layers_test tests/transformer_layers_test.cpp)
    add_executable(aabhas_transformer_block_test tests/transformer_block_test.cpp)
    add_executable(aabhas_quantized_kernels_test tests/quantized_kernels_test.cpp)
    target_compile_features(aabhas_model_index_test PRIVATE cxx_std_17)
    target_compile_features(aabhas_layer_scheduler_test PRIVATE cxx_std_17)
    target_compile_features(aabhas_inference_runtime_test PRIVATE cxx_std_17)
    target_compile_features(aabhas_tokenizer_test PRIVATE cxx_std_17)
    target_compile_features(aabhas_transformer_layers_test PRIVATE cxx_std_17)
    target_compile_features(aabhas_transformer_block_test PRIVATE cxx_std_17)
    target_compile_features(aabhas_quantized_kernels_test PRIVATE cxx_std_17)
    target_link_libraries(aabhas_model_index_test PRIVATE
        aabhas_data_plane
        Threads::Threads
    )
    target_link_libraries(aabhas_layer_scheduler_test PRIVATE
        aabhas_data_plane
        Threads::Threads
    )
    target_link_libraries(aabhas_inference_runtime_test PRIVATE
        aabhas_data_plane
        Threads::Threads
    )
    target_link_libraries(aabhas_tokenizer_test PRIVATE
        aabhas_data_plane
        Threads::Threads
    )
    target_link_libraries(aabhas_transformer_layers_test PRIVATE
        aabhas_data_plane
        Threads::Threads
    )
    target_link_libraries(aabhas_transformer_block_test PRIVATE
        aabhas_data_plane
        Threads::Threads
    )
    target_link_libraries(aabhas_quantized_kernels_test PRIVATE
        aabhas_data_plane
        Threads::Threads
    )
    add_test(NAME aabhas_model_index_test COMMAND aabhas_model_index_test)
    add_test(NAME aabhas_layer_scheduler_test COMMAND aabhas_layer_scheduler_test)
    add_test(NAME aabhas_inference_runtime_test COMMAND aabhas_inference_runtime_test)
    add_test(NAME aabhas_tokenizer_test COMMAND aabhas_tokenizer_test)
    add_test(NAME aabhas_transformer_layers_test COMMAND aabhas_transformer_layers_test)
    add_test(NAME aabhas_transformer_block_test COMMAND aabhas_transformer_block_test)
    add_test(NAME aabhas_quantized_kernels_test COMMAND aabhas_quantized_kernels_test)
endif()
