cmake_minimum_required(VERSION 3.24)
project(distributed_quadtree LANGUAGES CXX CUDA)

# ===========================================================================
# Monorepo root.
#
# This file owns the *shared* build configuration (language standards, GPU
# architectures, debug knobs, and the CUDA / NCCL package discovery). Each
# sub-project below only declares its own targets and reuses the function and
# imported targets defined here.
#
#   repulsive_forces/  distributed Barnes-Hut quad-tree (CUDA + NCCL)
#   attractive_forces/ (WIP) attractive-force half of the layout solver
# ===========================================================================

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

# Apply common debug/define options to a target. Defined at the monorepo root
# so every sub-project (repulsive_forces, attractive_forces, ...) can call it.
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
# manual find. Set NCCL_ROOT (or env NCCL_HOME) if installed in a non-standard
# path. The resulting NCCL::nccl imported target is created at the root so it is
# visible to every sub-project that links against it.
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
    add_library(NCCL::nccl UNKNOWN IMPORTED GLOBAL)
    set_target_properties(NCCL::nccl PROPERTIES
        IMPORTED_LOCATION "${NCCL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}")
    message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
endif()

enable_testing()

add_subdirectory(repulsive_forces)
add_subdirectory(attractive_forces)
