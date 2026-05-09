cmake_minimum_required(VERSION 3.24)
project(nccl_allreduce_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to Ampere (A100 / RTX 3090) + Hopper. Override with
# -DCMAKE_CUDA_ARCHITECTURES=... if you need something else.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 90)
endif()

find_package(CUDAToolkit REQUIRED)

# NCCL — try the CMake config first (recent NCCL ships one), fall back to a
# manual find. Set NCCL_ROOT (or env NCCL_HOME) if installed in a non-standard path.
find_package(NCCL CONFIG QUIET)
if(NOT NCCL_FOUND)
    find_path(NCCL_INCLUDE_DIR nccl.h
        HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT} $ENV{NCCL_HOME}
              ${CUDAToolkit_INCLUDE_DIRS}
        PATH_SUFFIXES include)
    find_library(NCCL_LIBRARY nccl
        HINTS ${NCCL_ROOT} $ENV{NCCL_ROOT} $ENV{NCCL_HOME}
              ${CUDAToolkit_LIBRARY_DIR}
        PATH_SUFFIXES lib lib64)
    if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
        message(FATAL_ERROR "NCCL not found. Set -DNCCL_ROOT=/path/to/nccl")
    endif()
    add_library(NCCL::nccl UNKNOWN IMPORTED)
    set_target_properties(NCCL::nccl PROPERTIES
        IMPORTED_LOCATION "${NCCL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}")
    message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
endif()

add_executable(main src/main.cu)
target_link_libraries(main
    PRIVATE
        NCCL::nccl
        CUDA::cudart)
target_include_directories(main PRIVATE include)

# Single-GPU smoke test for build_tree (no NCCL).
add_executable(test_build_tree src/test_build_tree.cu)
target_link_libraries(test_build_tree PRIVATE CUDA::cudart)
target_include_directories(test_build_tree PRIVATE include)

# Single-GPU smoke test for summarize_kernel (no NCCL).
add_executable(test_summarize src/test_summarize.cu)
target_link_libraries(test_summarize PRIVATE CUDA::cudart)
target_include_directories(test_summarize PRIVATE include)

# Single-GPU test for summarize_kernel that pretty-prints the whole tree.
add_executable(test_summarize_tree src/test_summarize_tree.cu)
target_link_libraries(test_summarize_tree PRIVATE CUDA::cudart)
target_include_directories(test_summarize_tree PRIVATE include)

# Two-GPU integration test for prepare_to_send_levels +
# apply_sumamary_across_nodes (uses NCCL).
add_executable(test_send_levels src/test_send_levels.cu)
target_link_libraries(test_send_levels
    PRIVATE
        NCCL::nccl
        CUDA::cudart)
target_include_directories(test_send_levels PRIVATE include)

# Single-GPU end-to-end test for the traverse_tree pipeline (no NCCL):
# build_tree -> summarize_kernel -> SortNodes -> traverse_tree.
add_executable(test_traverse_tree src/test_traverse_tree.cu)
target_link_libraries(test_traverse_tree PRIVATE CUDA::cudart)
target_include_directories(test_traverse_tree PRIVATE include)
