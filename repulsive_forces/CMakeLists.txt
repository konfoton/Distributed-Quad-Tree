# ===========================================================================
# repulsive_forces: distributed Barnes-Hut quad-tree (CUDA + NCCL).
#
# Shared configuration (standards, GPU arch, debug knobs, the
# apply_debug_options() helper, and the CUDA::cudart / NCCL::nccl imported
# targets) is defined in the monorepo root CMakeLists.txt and inherited here.
# ===========================================================================

add_executable(main src/main.cu src/kernels.cu)
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

# Full multi-GPU pipeline (4 GPUs, NCCL). No prints; built for profiling.
# bbox -> AllReduce -> build_tree -> summarize -> send_levels -> sort -> traverse.
add_executable(pipeline src/pipeline.cu src/kernels.cu)
target_link_libraries(pipeline
    PRIVATE
        NCCL::nccl
        CUDA::cudart)
target_include_directories(pipeline PRIVATE include)
apply_debug_options(pipeline)
