// Single-GPU smoke test for summarize_kernel (no NCCL).
//
// Builds a known level-1 tree (4 bodies, one per root quadrant -- no
// subdivides), runs summarize_kernel, and prints count_of_points and average
// for every populated cell.  The root cell holds all 4 bodies, so its
// expected count is 4 and its expected center of mass is (0, 0).
//
// Sizing convention (per user request):
//   average[]         is sized 2 * (number_of_points + number_of_cells)
//   count_of_points[] is sized     (number_of_points + number_of_cells)
// so that a body index in [0, number_of_points) and a cell index in
// [number_of_points, number_of_cells) both index the same flat arrays
// without overlap.
//
// Build & run:
//   cmake --build build --target test_summarize
//   ./build/test_summarize
//
// NOTE: summarize_kernel currently has known issues (see CLAUDE.md / chat):
//   - chx/chy declared inside an if-block, used outside     [compile error]
//   - coefficient declared inside a for-loop, used outside  [compile error]
//   - bottom = tree->number_of_free_cells (missing deref)   [logic bug]
// Until those are fixed this file will not link.  Once they are, this test
// is a quick way to verify the per-cell summary on a layout whose answer is
// trivial to check by hand.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "kernels.cuh"
#include "objects.cuh"

#define CUDA_CHECK(cmd)                                                        \
  do {                                                                         \
    cudaError_t e = (cmd);                                                     \
    if (e != cudaSuccess) {                                                    \
      std::fprintf(stderr, "CUDA error %s:%d '%s'\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(e));                                     \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

namespace {

// Build a tree from `host_points` and return:
//   d_points         - device pointer to the points array
//   d_tree           - device pointer to the populated tree struct
//   d_root           - device pointer to the root struct
//   d_free           - device pointer to the free-cell counter
// All three are owned by the caller (free with cudaFree).
struct DeviceTree {
  float* d_points;
  tree*  d_tree;
  root*  d_root;
  unsigned int* d_free;
  int*   d_cells;
  unsigned int max_cells;
  int    number_of_points;
};

DeviceTree build_known_tree(const std::vector<float>& host_points,
                            float root_x, float root_y, float root_r) {
  DeviceTree dt{};
  dt.number_of_points = static_cast<int>(host_points.size() / 2);
  dt.max_cells = 8u * static_cast<unsigned int>(dt.number_of_points);
  if (dt.max_cells < 16u) dt.max_cells = 16u;
  const size_t cells_array_len =
      static_cast<size_t>(dt.max_cells) * 4u;

  CUDA_CHECK(cudaMalloc(&dt.d_points,
                        sizeof(float) * host_points.size()));
  CUDA_CHECK(cudaMemcpy(dt.d_points, host_points.data(),
                        sizeof(float) * host_points.size(),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&dt.d_cells,
                        sizeof(int) * cells_array_len));
  CUDA_CHECK(cudaMalloc(&dt.d_free, sizeof(unsigned int)));

  std::vector<int>  init_cells(cells_array_len, -1);
  unsigned int      init_free = dt.max_cells - 1u;
  CUDA_CHECK(cudaMemcpy(dt.d_cells, init_cells.data(),
                        sizeof(int) * cells_array_len,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dt.d_free, &init_free, sizeof(unsigned int),
                        cudaMemcpyHostToDevice));

  tree h_tree{};
  h_tree.number_of_cells      = dt.max_cells;
  h_tree.number_of_free_cells = dt.d_free;
  h_tree.cells                = dt.d_cells;
  CUDA_CHECK(cudaMalloc(&dt.d_tree, sizeof(tree)));
  CUDA_CHECK(cudaMemcpy(dt.d_tree, &h_tree, sizeof(tree),
                        cudaMemcpyHostToDevice));

  root h_root{root_x, root_y, root_r};
  CUDA_CHECK(cudaMalloc(&dt.d_root, sizeof(root)));
  CUDA_CHECK(cudaMemcpy(dt.d_root, &h_root, sizeof(root),
                        cudaMemcpyHostToDevice));

  build_tree<<<1, 32>>>(dt.d_points, dt.number_of_points,
                        dt.d_tree, dt.d_root);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  return dt;
}

void free_device_tree(DeviceTree& dt) {
  CUDA_CHECK(cudaFree(dt.d_points));
  CUDA_CHECK(cudaFree(dt.d_cells));
  CUDA_CHECK(cudaFree(dt.d_free));
  CUDA_CHECK(cudaFree(dt.d_tree));
  CUDA_CHECK(cudaFree(dt.d_root));
}

}  // namespace

int main() {
  CUDA_CHECK(cudaSetDevice(0));

  // 4 well-separated bodies -- one per root quadrant of [-2, 2]^2.
  // Layout has body i at the corner of its quadrant; the centroid is (0, 0).
  std::vector<float> host_points = {
    -1.5f, -1.5f,  // body 0 -> SW
     1.5f, -1.5f,  // body 1 -> SE
    -1.5f,  1.5f,  // body 2 -> NW
     1.5f,  1.5f,  // body 3 -> NE
  };

  DeviceTree dt =
      build_known_tree(host_points, 0.0f, 0.0f, 2.0f);
  const int N = dt.number_of_points;
  const unsigned int max_cells = dt.max_cells;

  // -- summarize_kernel input arrays ----------------------------------------
  // Sized so both body and cell indices fall within bounds.
  const size_t com_len =
      static_cast<size_t>(N) + static_cast<size_t>(max_cells);

  float* d_average = nullptr;
  int*   d_count   = nullptr;
  CUDA_CHECK(cudaMalloc(&d_average, sizeof(float) * 2 * com_len));
  CUDA_CHECK(cudaMalloc(&d_count,   sizeof(int)   * com_len));

  // count_of_points = -1 everywhere is the "not yet processed" sentinel.
  std::vector<int>   init_count(com_len, -1);
  std::vector<float> init_avg(com_len * 2, 0.0f);
  CUDA_CHECK(cudaMemcpy(d_count, init_count.data(),
                        sizeof(int) * com_len, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_average, init_avg.data(),
                        sizeof(float) * 2 * com_len,
                        cudaMemcpyHostToDevice));

  summarize_kernel<<<1, 32>>>(dt.d_points, d_average, d_count, dt.d_tree,
                              N, static_cast<int>(max_cells));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // -- read back & dump -----------------------------------------------------
  std::vector<int>   out_count(com_len);
  std::vector<float> out_avg(com_len * 2);
  CUDA_CHECK(cudaMemcpy(out_count.data(), d_count,
                        sizeof(int) * com_len, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out_avg.data(), d_average,
                        sizeof(float) * 2 * com_len,
                        cudaMemcpyDeviceToHost));

  std::vector<int> out_cells(static_cast<size_t>(max_cells) * 4u);
  CUDA_CHECK(cudaMemcpy(out_cells.data(), dt.d_cells,
                        sizeof(int) * out_cells.size(),
                        cudaMemcpyDeviceToHost));

  std::printf("\n==== tree layout (root = cell %u) ====\n", max_cells - 1u);
  for (unsigned int c = 0; c < max_cells; ++c) {
    int v0 = out_cells[c * 4 + 0];
    int v1 = out_cells[c * 4 + 1];
    int v2 = out_cells[c * 4 + 2];
    int v3 = out_cells[c * 4 + 3];
    if (v0 != -1 || v1 != -1 || v2 != -1 || v3 != -1) {
      std::printf(
          "cell %2u: SW=%3d  SE=%3d  NW=%3d  NE=%3d\n",
          c, v0, v1, v2, v3);
    }
  }

  std::printf("\n==== summarize_kernel output ====\n");
  std::printf("(only entries with count >= 0 are shown)\n");
  for (size_t i = 0; i < com_len; ++i) {
    if (out_count[i] >= 0) {
      std::printf("idx %3zu: count=%4d  average=(% .4f, % .4f)\n",
                  i, out_count[i],
                  out_avg[i * 2], out_avg[i * 2 + 1]);
    }
  }

  // -- correctness: root cell must hold all N bodies, centroid (0,0) ---------
  const int root_idx = static_cast<int>(max_cells) - 1;
  const int   root_count = out_count[root_idx];
  const float root_ax    = out_avg[root_idx * 2];
  const float root_ay    = out_avg[root_idx * 2 + 1];

  std::printf("\n==== root check (cell %d) ====\n", root_idx);
  std::printf("count_of_points[root] = %d   (expected %d)\n", root_count, N);
  std::printf("average[root]         = (% .4f, % .4f)   (expected ( 0.0000,  0.0000))\n",
              root_ax, root_ay);

  bool ok = (root_count == N) &&
            (std::fabs(root_ax) < 1e-3f) &&
            (std::fabs(root_ay) < 1e-3f);
  std::printf("\nroot summary: %s\n", ok ? "OK" : "FAIL");

  CUDA_CHECK(cudaFree(d_average));
  CUDA_CHECK(cudaFree(d_count));
  free_device_tree(dt);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
