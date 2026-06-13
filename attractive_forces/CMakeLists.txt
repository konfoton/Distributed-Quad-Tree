# ===========================================================================
# attractive_forces: (WIP) attractive-force half of the layout solver.
#
# Currently just a dummy SAXPY program to prove out the monorepo build. Shared
# configuration (standards, GPU arch, debug knobs, apply_debug_options(), and
# the CUDA::cudart imported target) comes from the monorepo root CMakeLists.txt.
# ===========================================================================

add_executable(attractive_main src/main.cu src/kernels.cu)
target_link_libraries(attractive_main PRIVATE CUDA::cudart)
target_include_directories(attractive_main PRIVATE include)
apply_debug_options(attractive_main)
