// Single-GPU end-to-end test for the traverse_tree pipeline (no NCCL).
//
// Wires the full body-force pipeline together on a small deterministic point
// cloud:
//   build_tree           -> populate tree->cells from points
//   clear_kernel_two     -> reset count_of_points to -1 sentinel
//   summarize_kernel     -> per-cell count + centroid (center of mass)
//   ClearKernelthree    -> reset cumulative-count buffer used by SortNodes
//   SortNodes            -> in-order body permutation grouped by cell
//   traverse_tree        -> Barnes-Hut force accumulation per body
//
// Then we copy the sorted permutation and the per-body gradient back to the
// host and print them.  Reachability of the sort (every body listed exactly
// once) is checked as the cheap correctness signal.
//
// Build & run:
//   cmake --build build --target test_traverse_tree
//   ./build/test_traverse_tree
//
// NOTE: traverse_tree, SortNodes, ClearKernelthree are work-in-progress (see
// CLAUDE.md). This test compiles them and exercises the data flow; it does not
// validate that the gradient values are physically correct.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "bounding_box.cuh"
#include "objects.cuh"

#define CUDA_CHECK(cmd)                                                   \
  do {                                                                    \
    cudaError_t e = (cmd);                                                \
    if (e != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                                \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

namespace {

float lcg(std::uint32_t& state) {
  state = state * 1664525u + 1013904223u;
  const float u = (state >> 8) * (1.0f / 16777216.0f);
  return 2.0f * u - 1.0f;
}

}  // namespace

int main() {
  CUDA_CHECK(cudaSetDevice(0));

  // ---- 0. host-side input ------------------------------------------------
  const int number_of_points = 16;
  const float root_x = 0.0f, root_y = 0.0f, root_r = 2.0f;

  std::vector<float> host_points;
  host_points.reserve(number_of_points * 2);
  std::uint32_t s = 0xCAFEBABEu;
  for (int i = 0; i < number_of_points; ++i) {
    host_points.push_back(lcg(s) * 1.5f);
    host_points.push_back(lcg(s) * 1.5f);
  }

  std::printf("==== input points (n=%d) ====\n", number_of_points);
  for (int i = 0; i < number_of_points; ++i) {
    std::printf("  body %2d: (% .4f, % .4f)\n", i,
                host_points[i * 2], host_points[i * 2 + 1]);
  }

  // ---- 1. tree storage ---------------------------------------------------
  unsigned int max_cells = 8u * static_cast<unsigned int>(number_of_points);
  if (max_cells < 64u) max_cells = 64u;
  const size_t cells_array_len = static_cast<size_t>(max_cells) * 4u;
  const size_t com_len =
      static_cast<size_t>(number_of_points) + max_cells;

  float* d_points = nullptr;
  CUDA_CHECK(cudaMalloc(&d_points, sizeof(float) * host_points.size()));
  CUDA_CHECK(cudaMemcpy(d_points, host_points.data(),
                        sizeof(float) * host_points.size(),
                        cudaMemcpyHostToDevice));

  int* d_cells = nullptr;
  unsigned int* d_free = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cells, sizeof(int) * cells_array_len));
  CUDA_CHECK(cudaMalloc(&d_free, sizeof(unsigned int)));

  std::vector<int> init_cells(cells_array_len, -1);
  unsigned int init_free = max_cells - 1u;
  CUDA_CHECK(cudaMemcpy(d_cells, init_cells.data(),
                        sizeof(int) * cells_array_len, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_free, &init_free, sizeof(unsigned int),
                        cudaMemcpyHostToDevice));

  tree h_tree{};
  h_tree.number_of_cells = max_cells;
  h_tree.number_of_free_cells = d_free;
  h_tree.cells = d_cells;
  tree* d_tree = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tree, sizeof(tree)));
  CUDA_CHECK(cudaMemcpy(d_tree, &h_tree, sizeof(tree), cudaMemcpyHostToDevice));

  root h_root{root_x, root_y, root_r};
  root* d_root = nullptr;
  CUDA_CHECK(cudaMalloc(&d_root, sizeof(root)));
  CUDA_CHECK(cudaMemcpy(d_root, &h_root, sizeof(root), cudaMemcpyHostToDevice));

  // ---- 2. build_tree -----------------------------------------------------
  build_tree<<<1, 64>>>(d_points, number_of_points, d_tree, d_root);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- 3. clear_kernel_two + summarize_kernel ----------------------------
  float* d_average = nullptr;
  int* d_count_of_points = nullptr;
  CUDA_CHECK(cudaMalloc(&d_average, sizeof(float) * 2 * com_len));
  CUDA_CHECK(cudaMalloc(&d_count_of_points, sizeof(int) * com_len));

  clear_kernel_two<<<1, 256>>>(d_average, d_count_of_points,
                               static_cast<int>(com_len));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  summarize_kernel<<<1, 32>>>(d_points, d_average, d_count_of_points, d_tree,
                              number_of_points, static_cast<int>(max_cells));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- 4. ClearKernelthree + SortNodes -----------------------------------
  // count[k] holds the cumulative starting offset (in `sorted`) of the bodies
  // under cell k. SortNodes propagates it top-down.
  int* d_count = nullptr;
  int* d_sorted = nullptr;
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(int) * com_len));
  CUDA_CHECK(cudaMalloc(&d_sorted, sizeof(int) * number_of_points));

  ClearKernelthree<<<1, 256>>>(d_count, d_tree);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  SortNodes<<<1, 256>>>(d_count, d_sorted, d_points, d_count_of_points, d_tree,
                        number_of_points, static_cast<int>(max_cells));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- 5. traverse_tree --------------------------------------------------
  float* d_gradient = nullptr;
  CUDA_CHECK(cudaMalloc(&d_gradient, sizeof(float) * 2 * number_of_points));
  CUDA_CHECK(cudaMemset(d_gradient, 0, sizeof(float) * 2 * number_of_points));

  const float theta = 0.5f;
  const float itolsqd = 1.0f / (theta * theta);
  const float epssqd = 1e-3f;

  // traverse_tree's signature declares count_of_points as float* (treated as
  // a per-cell weight). We pass the int* buffer reinterpreted; matches the
  // current kernel signature so the test compiles.
  traverse_tree<<<1, 32>>>(d_tree, d_root, itolsqd, epssqd, d_sorted, d_average,
                           reinterpret_cast<float*>(d_count_of_points),
                           static_cast<int>(max_cells), number_of_points,
                           d_points, d_gradient);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- 6. read back & dump ----------------------------------------------
  std::vector<int> out_sorted(number_of_points);
  std::vector<float> out_gradient(2 * number_of_points);
  std::vector<int> out_count_of_points(com_len);
  CUDA_CHECK(cudaMemcpy(out_sorted.data(), d_sorted,
                        sizeof(int) * number_of_points,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out_gradient.data(), d_gradient,
                        sizeof(float) * 2 * number_of_points,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out_count_of_points.data(), d_count_of_points,
                        sizeof(int) * com_len, cudaMemcpyDeviceToHost));

  unsigned int out_free = 0;
  CUDA_CHECK(cudaMemcpy(&out_free, d_free, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));
  std::printf("\nfree pool head: start=%u  end=%u  (cells consumed=%d)\n",
              init_free, out_free,
              static_cast<int>(init_free) - static_cast<int>(out_free));

  const int root_idx = static_cast<int>(max_cells) - 1;
  std::printf("root cell %d: count=%d\n", root_idx,
              out_count_of_points[root_idx]);

  std::printf("\n==== sorted permutation (length %d) ====\n", number_of_points);
  for (int k = 0; k < number_of_points; ++k) {
    const int b = out_sorted[k];
    if (b >= 0 && b < number_of_points) {
      std::printf("  sorted[%2d] = body %2d  @ (% .4f, % .4f)\n", k, b,
                  host_points[b * 2], host_points[b * 2 + 1]);
    } else {
      std::printf("  sorted[%2d] = %d  (out of range)\n", k, b);
    }
  }

  // Reachability check: every body must appear exactly once in the sort.
  std::vector<int> seen(number_of_points, 0);
  bool sort_ok = true;
  for (int k = 0; k < number_of_points; ++k) {
    const int b = out_sorted[k];
    if (b < 0 || b >= number_of_points || ++seen[b] > 1) {
      sort_ok = false;
    }
  }
  std::printf("\nsort reachability: each body appears once -> %s\n",
              sort_ok ? "OK" : "FAIL");

  std::printf("\n==== per-body gradient (indexed by sort position k) ====\n");
  for (int k = 0; k < number_of_points; ++k) {
    std::printf("  k=%2d  body=%2d  grad=(% .6f, % .6f)\n", k, out_sorted[k],
                out_gradient[k * 2], out_gradient[k * 2 + 1]);
  }

  // ---- cleanup -----------------------------------------------------------
  CUDA_CHECK(cudaFree(d_points));
  CUDA_CHECK(cudaFree(d_cells));
  CUDA_CHECK(cudaFree(d_free));
  CUDA_CHECK(cudaFree(d_tree));
  CUDA_CHECK(cudaFree(d_root));
  CUDA_CHECK(cudaFree(d_average));
  CUDA_CHECK(cudaFree(d_count_of_points));
  CUDA_CHECK(cudaFree(d_count));
  CUDA_CHECK(cudaFree(d_sorted));
  CUDA_CHECK(cudaFree(d_gradient));

  std::printf("\ndone.\n");
  return sort_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
