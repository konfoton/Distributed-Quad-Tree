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

# Default to Release if the user didn't pick a build type. Override with
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --------------------------------------------------------------------------
# Debug knobs (override at configure time)
#
#   -DCUDA_DEVICE_DEBUG=ON      : add `-G -g` so cuda-gdb / compute-sanitizer
#                                 can step into device code (also implied by
#                                 -DCMAKE_BUILD_TYPE=Debug).
#   -DCUDA_LINEINFO=ON          : add `-lineinfo` (cheap; nsight-compute
#                                 source mapping without disabling opts).
#   -DEXTRA_DEFINES="A=1;B=2"   : preprocessor defines applied to every
#                                 target. Single value? -DEXTRA_DEFINES=FOO=1.
#                                 Use them in code with `#ifdef FOO` etc.
#   -DEXTRA_CUDA_FLAGS="..."    : raw flags appended to nvcc invocations
#                                 (space-separated string).
# --------------------------------------------------------------------------
option(CUDA_DEVICE_DEBUG "Compile device code with -G (full device debug)" OFF)
option(CUDA_LINEINFO     "Compile device code with -lineinfo"              OFF)
set(EXTRA_DEFINES   "" CACHE STRING "Extra preprocessor defines (semicolon-separated, e.g. FOO=1;BAR)")
set(EXTRA_CUDA_FLAGS "" CACHE STRING "Extra raw flags forwarded to nvcc (space-separated)")

# Apply common debug/define options to a target.
function(apply_debug_options target)
    if(CUDA_DEVICE_DEBUG OR CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-G>
            $<$<COMPILE_LANGUAGE:CUDA>:-g>)
    elseif(CUDA_LINEINFO OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    endif()

    if(EXTRA_DEFINES)
        target_compile_definitions(${target} PRIVATE ${EXTRA_DEFINES})
    endif()

    if(EXTRA_CUDA_FLAGS)
        # Split the user-supplied space-separated string so each flag is its
        # own argv element (CMake otherwise quotes it as one big arg).
        separate_arguments(_extra_nvcc UNIX_COMMAND "${EXTRA_CUDA_FLAGS}")
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${_extra_nvcc}>)
    endif()
endfunction()

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
apply_debug_options(main)

# Single-GPU smoke test for build_tree (no NCCL).
add_executable(test_build_tree src/test_build_tree.cu src/kernels.cu)
target_link_libraries(test_build_tree PRIVATE CUDA::cudart)
target_include_directories(test_build_tree PRIVATE include)
apply_debug_options(test_build_tree)

# Single-GPU smoke test for summarize_kernel (no NCCL).
add_executable(test_summarize src/test_summarize.cu src/kernels.cu)
target_link_libraries(test_summarize PRIVATE CUDA::cudart)
target_include_directories(test_summarize PRIVATE include)
apply_debug_options(test_summarize)

# Single-GPU test for summarize_kernel that pretty-prints the whole tree.
add_executable(test_summarize_tree src/test_summarize_tree.cu src/kernels.cu)
target_link_libraries(test_summarize_tree PRIVATE CUDA::cudart)
target_include_directories(test_summarize_tree PRIVATE include)
apply_debug_options(test_summarize_tree)

# Two-GPU integration test for prepare_to_send_levels +
# apply_sumamary_across_nodes (uses NCCL).
add_executable(test_send_levels src/test_send_levels.cu src/kernels.cu)
target_link_libraries(test_send_levels
    PRIVATE
        NCCL::nccl
        CUDA::cudart)
target_include_directories(test_send_levels PRIVATE include)
apply_debug_options(test_send_levels)

# Single-GPU end-to-end test for the traverse_tree pipeline (no NCCL):
# build_tree -> summarize_kernel -> SortNodes -> traverse_tree.
add_executable(test_traverse_tree src/test_traverse_tree.cu src/kernels.cu) 
target_link_libraries(test_traverse_tree PRIVATE CUDA::cudart)
target_include_directories(test_traverse_tree PRIVATE include)
apply_debug_options(test_traverse_tree)
