cmake_minimum_required(VERSION 3.16)

project(aabhas_data_plane LANGUAGES CXX)

add_library(aabhas_data_plane STATIC
    src/memory/mmap_illusion.cpp
    src/hardware/nvme_streamer.cpp
    src/hardware/hw_identifier.cpp
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
