// Single-GPU test for `summarize_kernel` that pretty-prints the whole tree.
//
// 30 deterministic points are scattered across [-1.5, 1.5]^2; the root is
// (0, 0) radius 2. We run build_tree + summarize_kernel and walk the cells
// array recursively, printing every populated node (cells annotated with
// count + centroid from summarize_kernel; bodies with index + position).
//
// Build & run:
//   cmake --build build --target test_summarize_tree
//   ./build/test_summarize_tree

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "kernels.cuh"
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

const char* const kQuadrantName[4] = {"SW", "SE", "NW", "NE"};
const float kDx[4] = {-1.0f, +1.0f, -1.0f, +1.0f};
const float kDy[4] = {-1.0f, -1.0f, +1.0f, +1.0f};

void print_subtree(const std::vector<int>& cells,
                   const std::vector<float>& points,
                   const std::vector<int>& count,
                   const std::vector<float>& avg,
                   int number_of_points,
                   int cell_idx, float cx, float cy, float r,
                   const std::string& prefix) {
  const float child_r = r * 0.5f;
  for (int s = 0; s < 4; ++s) {
    const bool last = (s == 3);
    const std::string branch     = prefix + (last ? "`-- " : "|-- ");
    const std::string sub_prefix = prefix + (last ? "    " : "|   ");

    const int v   = cells[cell_idx * 4 + s];
    const float ccx = cx + kDx[s] * child_r;
    const float ccy = cy + kDy[s] * child_r;

    if (v == -1) {
      std::printf("%s%s empty\n", branch.c_str(), kQuadrantName[s]);
    } else if (v == -2) {
      std::printf("%s%s LOCKED\n", branch.c_str(), kQuadrantName[s]);
    } else if (v < number_of_points) {
      std::printf("%s%s body %d @ (% .3f, % .3f)\n",
                  branch.c_str(), kQuadrantName[s], v,
                  points[v * 2], points[v * 2 + 1]);
    } else {
      const int   c  = (v < (int)count.size()) ? count[v]         : -1;
      const float ax = (v < (int)count.size()) ? avg[v * 2]       : 0.f;
      const float ay = (v < (int)count.size()) ? avg[v * 2 + 1]   : 0.f;
      std::printf("%s%s cell %d  count=%d  centroid=(% .3f, % .3f)  "
                  "(geom center=(% .3f, % .3f) r=%.3f)\n",
                  branch.c_str(), kQuadrantName[s], v, c, ax, ay,
                  ccx, ccy, child_r);
      print_subtree(cells, points, count, avg, number_of_points, v,
                    ccx, ccy, child_r, sub_prefix);
    }
  }
}

void print_tree(const std::vector<int>& cells,
                const std::vector<float>& points,
                const std::vector<int>& count,
                const std::vector<float>& avg,
                int number_of_points,
                int root_idx, float cx, float cy, float r) {
  const int   c  = (root_idx < (int)count.size()) ? count[root_idx]       : -1;
  const float ax = (root_idx < (int)count.size()) ? avg[root_idx * 2]     : 0.f;
  const float ay = (root_idx < (int)count.size()) ? avg[root_idx * 2 + 1] : 0.f;
  std::printf("[root cell %d]  count=%d  centroid=(% .3f, % .3f)  "
              "(geom center=(% .3f, % .3f) r=%.3f)\n",
              root_idx, c, ax, ay, cx, cy, r);
  print_subtree(cells, points, count, avg, number_of_points, root_idx,
                cx, cy, r, std::string());
}

// Deterministic LCG -> uniform in [-1, 1).
float lcg(std::uint32_t& state) {
  state = state * 1664525u + 1013904223u;
  const float u = (state >> 8) * (1.0f / 16777216.0f);
  return 2.0f * u - 1.0f;
}

}  // namespace

int main() {
  CUDA_CHECK(cudaSetDevice(0));

  // ---- 0. host-side input ------------------------------------------------
  const int number_of_points = 30;
  const float root_x = 0.0f, root_y = 0.0f, root_r = 2.0f;

  std::vector<float> host_points;
  host_points.reserve(number_of_points * 2);
  std::uint32_t s = 0xDEADBEEFu;
  for (int i = 0; i < number_of_points; ++i) {
    // sample in [-1.5, 1.5]^2 so points sit comfortably inside the root.
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

  int*          d_cells = nullptr;
  unsigned int* d_free  = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cells, sizeof(int) * cells_array_len));
  CUDA_CHECK(cudaMalloc(&d_free,  sizeof(unsigned int)));

  std::vector<int> init_cells(cells_array_len, -1);
  unsigned int     init_free = max_cells - 1u;
  CUDA_CHECK(cudaMemcpy(d_cells, init_cells.data(),
                        sizeof(int) * cells_array_len, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_free, &init_free, sizeof(unsigned int),
                        cudaMemcpyHostToDevice));

  tree h_tree{};
  h_tree.number_of_cells      = max_cells;
  h_tree.number_of_free_cells = d_free;
  h_tree.cells                = d_cells;
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

  // ---- 3. summarize_kernel ----------------------------------------------
  float* d_average = nullptr;
  int*   d_count   = nullptr;
  CUDA_CHECK(cudaMalloc(&d_average, sizeof(float) * 2 * com_len));
  CUDA_CHECK(cudaMalloc(&d_count,   sizeof(int)   * com_len));

  std::vector<int>   init_count(com_len, -1);
  std::vector<float> init_avg(com_len * 2, 0.0f);
  CUDA_CHECK(cudaMemcpy(d_count, init_count.data(),
                        sizeof(int) * com_len, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_average, init_avg.data(),
                        sizeof(float) * 2 * com_len, cudaMemcpyHostToDevice));

  summarize_kernel<<<1, 32>>>(d_points, d_average, d_count, d_tree,
                              number_of_points, static_cast<int>(max_cells));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // ---- 4. read back & pretty-print --------------------------------------
  std::vector<int>   out_cells(cells_array_len);
  std::vector<int>   out_count(com_len);
  std::vector<float> out_avg(com_len * 2);
  CUDA_CHECK(cudaMemcpy(out_cells.data(), d_cells,
                        sizeof(int) * cells_array_len, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out_count.data(), d_count,
                        sizeof(int) * com_len, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out_avg.data(), d_average,
                        sizeof(float) * 2 * com_len, cudaMemcpyDeviceToHost));
  unsigned int out_free = 0;
  CUDA_CHECK(cudaMemcpy(&out_free, d_free, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));

  std::printf("\nfree pool head: start=%u  end=%u  (cells consumed=%d)\n",
              init_free, out_free,
              static_cast<int>(init_free) - static_cast<int>(out_free));

  const int root_idx = static_cast<int>(max_cells) - 1;
  std::printf("\n==== tree (root=cell %d, geom center=(%.3f, %.3f), r=%.3f) "
              "====\n", root_idx, root_x, root_y, root_r);
  print_tree(out_cells, host_points, out_count, out_avg,
             number_of_points, root_idx, root_x, root_y, root_r);

  // ---- cleanup -----------------------------------------------------------
  CUDA_CHECK(cudaFree(d_points));
  CUDA_CHECK(cudaFree(d_cells));
  CUDA_CHECK(cudaFree(d_free));
  CUDA_CHECK(cudaFree(d_tree));
  CUDA_CHECK(cudaFree(d_root));
  CUDA_CHECK(cudaFree(d_average));
  CUDA_CHECK(cudaFree(d_count));

  std::printf("\ndone.\n");
  return 0;
}
