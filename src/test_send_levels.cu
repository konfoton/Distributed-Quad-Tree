// Two-GPU integration test for `prepare_to_send_levels` and
// `apply_sumamary_across_nodes` (note: the kernel name in
// bounding_box.cuh has a typo -- "sumamary" instead of "summary").
//
// Each GPU is given its OWN, DIFFERENT, large-ish point cloud (a couple
// hundred points each, sampled deterministically with a tiny LCG so the
// run is reproducible). Pipeline exercised here, end to end on 2 GPUs:
//
//   1. calculate_bounding_box          (per-GPU)
//   2. ncclAllReduce min/min/max/max   -> global bbox on every GPU
//   3. derive a single `root` from the global bbox; same root on both GPUs
//   4. build_tree                      (per-GPU)
//   5. summarize_kernel                (per-GPU; centroid + count per cell)
//   6. prepare_to_send_levels          (per-GPU; pack top k+1 levels)
//   7. ncclAllReduce SUM on the packed buffer
//   8. apply_sumamary_across_nodes     (per-GPU; scatter back into tree)
//
// This is a scaffold -- it does NOT return a pass/fail; it just prints
// the per-GPU tree at every interesting stage so the result is easy to
// eyeball. (The two `prepare_to_send_levels` / `apply_sumamary_across_nodes`
// kernels in bounding_box.cuh currently have known issues; this file is
// written so it starts running as soon as those bugs are fixed.)
//
// Build & run:
//   cmake --build build --target test_send_levels
//   ./build/test_send_levels      # requires >=2 CUDA GPUs and NCCL

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

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

#define NCCL_CHECK(cmd)                                                   \
  do {                                                                    \
    ncclResult_t r = (cmd);                                               \
    if (r != ncclSuccess) {                                               \
      std::fprintf(stderr, "NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
                   ncclGetErrorString(r));                                \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

namespace {

// ---- Pretty-printing helpers (extended from test_build_tree.cu) ---------
const char* const kQuadrantName[4] = {"SW", "SE", "NW", "NE"};
const float kDx[4] = {-1.0f, +1.0f, -1.0f, +1.0f};
const float kDy[4] = {-1.0f, -1.0f, +1.0f, +1.0f};

// Walk the tree and pretty-print each populated cell with the centroid +
// count produced by summarize_kernel. `max_depth` caps the recursion so a
// big point cloud doesn't dump thousands of lines.
void print_subtree(const std::vector<int>& cells,
                   const std::vector<float>& points,
                   const std::vector<int>& count,
                   const std::vector<float>& avg,
                   int number_of_points,
                   int cell_idx, float cx, float cy, float r,
                   const std::string& prefix,
                   int depth, int max_depth) {
  if (depth >= max_depth) {
    std::printf("%s... (depth cap reached)\n", prefix.c_str());
    return;
  }

  const float child_r = r * 0.5f;
  for (int s = 0; s < 4; ++s) {
    const bool last = (s == 3);
    const std::string branch = prefix + (last ? "`-- " : "|-- ");
    const std::string sub_prefix = prefix + (last ? "    " : "|   ");

    const int v = cells[cell_idx * 4 + s];
    const float ccx = cx + kDx[s] * child_r;
    const float ccy = cy + kDy[s] * child_r;

    if (v == -1) {
      // skip empty quadrants for compactness; flip this `continue` to
      // a printf if you want them shown.
      continue;
    } else if (v == -2) {
      std::printf("%s%s LOCKED\n", branch.c_str(), kQuadrantName[s]);
    } else if (v < number_of_points) {
      std::printf("%s%s body %d @ (%.3f, %.3f)\n", branch.c_str(),
                  kQuadrantName[s], v, points[v * 2], points[v * 2 + 1]);
    } else {
      const int    c   = (v < (int)count.size())     ? count[v] : -1;
      const float ax = (v < (int)count.size())     ? avg[v * 2]     : 0.f;
      const float ay = (v < (int)count.size())     ? avg[v * 2 + 1] : 0.f;
      std::printf("%s%s cell %d  count=%d  centroid=(% .3f, % .3f)  "
                  "(geom center=(% .3f, % .3f) r=%.3f)\n",
                  branch.c_str(), kQuadrantName[s], v, c, ax, ay,
                  ccx, ccy, child_r);
      print_subtree(cells, points, count, avg, number_of_points, v,
                    ccx, ccy, child_r, sub_prefix, depth + 1, max_depth);
    }
  }
}

void print_tree(const std::vector<int>& cells,
                const std::vector<float>& points,
                const std::vector<int>& count,
                const std::vector<float>& avg,
                int number_of_points,
                int root_idx, float cx, float cy, float r,
                int max_depth) {
  const int   c   = (root_idx < (int)count.size())     ? count[root_idx] : -1;
  const float ax = (root_idx < (int)count.size())     ? avg[root_idx * 2]     : 0.f;
  const float ay = (root_idx < (int)count.size())     ? avg[root_idx * 2 + 1] : 0.f;
  std::printf("[root cell %d]  count=%d  centroid=(% .3f, % .3f)  "
              "(geom center=(% .3f, % .3f) r=%.3f)\n",
              root_idx, c, ax, ay, cx, cy, r);
  print_subtree(cells, points, count, avg, number_of_points, root_idx,
                cx, cy, r, std::string(), 0, max_depth);
}

// Tiny deterministic LCG so each GPU's point cloud is different but
// reproducible across runs. Returns a float in [-1, 1).
float lcg(std::uint32_t& state) {
  state = state * 1664525u + 1013904223u;
  // top 24 bits -> [0, 1), then map to [-1, 1)
  const float u = (state >> 8) * (1.0f / 16777216.0f);
  return 2.0f * u - 1.0f;
}

// Per-GPU state -- every pointer is a *device* pointer on `dev`.
struct GpuState {
  int dev;

  node*  d_node;
  plane* d_plane;

  int*          d_cells;
  unsigned int* d_free;
  tree*         d_tree;
  root*         d_root;
  float*        d_points;
  float*        d_average;
  int*          d_count;

  float* d_result_average;     // size 2 * M
  int*   d_result_count;       // size M, int to match kernel signature

  cudaStream_t stream;

  int          number_of_points;
  unsigned int max_cells;
  size_t       com_len;
};

int compute_M(int number_of_layers) {
  int M = 0;
  int term = 1;
  for (int j = 0; j <= number_of_layers; ++j) {
    M += term;
    term *= 4;
  }
  return M;
}

}  // namespace

int main() {
  constexpr int nDev = 2;
  std::vector<int> devs = {0, 1};

  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < nDev) {
    std::fprintf(stderr, "Need %d GPUs, found %d\n", nDev, deviceCount);
    return EXIT_FAILURE;
  }

  // ---- 0. host-side test inputs ------------------------------------------
  // Two clearly-different point clouds, ~200 points each. GPU 0's cloud is
  // biased toward the SW quadrant; GPU 1's cloud is biased toward the NE
  // quadrant. After the cross-node summary, the *root* centroid on each
  // GPU should be the average of all 400 points -- somewhere near (0, 0).
  const int number_of_points = 200;

  std::vector<std::vector<float>> host_points(nDev);
  {
    std::uint32_t s0 = 0xC0FFEEu;
    host_points[0].reserve(number_of_points * 2);
    for (int i = 0; i < number_of_points; ++i) {
      // sampled in [-1, 0) x [-1, 0) -- biased SW
      host_points[0].push_back(lcg(s0) * 0.5f - 0.5f);
      host_points[0].push_back(lcg(s0) * 0.5f - 0.5f);
    }

    std::uint32_t s1 = 0xBADF00Du;
    host_points[1].reserve(number_of_points * 2);
    for (int i = 0; i < number_of_points; ++i) {
      // sampled in [0, 1) x [0, 1) -- biased NE
      host_points[1].push_back(lcg(s1) * 0.5f + 0.5f);
      host_points[1].push_back(lcg(s1) * 0.5f + 0.5f);
    }
  }

  const int number_of_threads = 256;
  const int number_of_blocks =
      (number_of_points + number_of_threads - 1) / number_of_threads;

  // Pack the top 2 levels (root + its 4 kids) -> 4^1 threads, M = 5.
  // Bump to 2 if you want root + 4 + 16 (= 21 slots).
  const int number_of_layers = 1;
  const int number_of_iter   = 1 << (2 * number_of_layers);
  const int M                = compute_M(number_of_layers);

  const int print_max_depth  = 4;  // cap the tree pretty-print

  // ---- 1-2. NCCL init + per-GPU buffers ----------------------------------
  std::vector<GpuState> gpus(nDev);
  std::vector<ncclComm_t> comms(nDev);
  NCCL_CHECK(ncclCommInitAll(comms.data(), nDev, devs.data()));

  builder creator;

  for (int i = 0; i < nDev; ++i) {
    GpuState& g = gpus[i];
    g.dev = devs[i];
    g.number_of_points = number_of_points;
    // 8 * N is the usual Barnes-Hut heuristic; with random points 200
    // bodies fit easily.
    g.max_cells = 8u * static_cast<unsigned int>(number_of_points);
    g.com_len   = static_cast<size_t>(number_of_points) + g.max_cells;

    CUDA_CHECK(cudaSetDevice(g.dev));
    CUDA_CHECK(cudaStreamCreate(&g.stream));

    g.d_node  = creator.create_node(host_points[i].data(), number_of_points);
    g.d_plane = creator.create_plane(number_of_blocks);

    CUDA_CHECK(cudaMalloc(&g.d_points,
                          sizeof(float) * host_points[i].size()));
    CUDA_CHECK(cudaMemcpy(g.d_points, host_points[i].data(),
                          sizeof(float) * host_points[i].size(),
                          cudaMemcpyHostToDevice));

    const size_t cells_array_len =
        static_cast<size_t>(g.max_cells) * 4u;
    CUDA_CHECK(cudaMalloc(&g.d_cells, sizeof(int) * cells_array_len));
    CUDA_CHECK(cudaMalloc(&g.d_free,  sizeof(unsigned int)));

    std::vector<int> init_cells(cells_array_len, -1);
    unsigned int     init_free  = g.max_cells - 1u;
    CUDA_CHECK(cudaMemcpy(g.d_cells, init_cells.data(),
                          sizeof(int) * cells_array_len,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_free, &init_free, sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    tree h_tree{};
    h_tree.number_of_cells      = g.max_cells;
    h_tree.number_of_free_cells = g.d_free;
    h_tree.cells                = g.d_cells;
    CUDA_CHECK(cudaMalloc(&g.d_tree, sizeof(tree)));
    CUDA_CHECK(cudaMemcpy(g.d_tree, &h_tree, sizeof(tree),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&g.d_root, sizeof(root)));

    CUDA_CHECK(cudaMalloc(&g.d_average, sizeof(float) * 2 * g.com_len));
    CUDA_CHECK(cudaMalloc(&g.d_count,   sizeof(int)   * g.com_len));

    CUDA_CHECK(cudaMalloc(&g.d_result_average, sizeof(float) * 2 * M));
    CUDA_CHECK(cudaMalloc(&g.d_result_count,   sizeof(int)   * M));
  }

  // ---- 1. calculate_bounding_box (per GPU) -------------------------------
  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(gpus[i].dev));
    calculate_bounding_box<<<number_of_blocks, number_of_threads>>>(
        gpus[i].d_node, gpus[i].d_plane);
  }
  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(gpus[i].dev));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // ---- 2. ncclAllReduce on the per-GPU bboxes ----------------------------
  std::vector<plane> planes_h(nDev);
  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(gpus[i].dev));
    CUDA_CHECK(cudaMemcpy(&planes_h[i], gpus[i].d_plane, sizeof(plane),
                          cudaMemcpyDeviceToHost));
  }

  NCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(gpus[i].dev));
    NCCL_CHECK(ncclAllReduce(planes_h[i].minx, planes_h[i].minx, 1,
                             ncclFloat, ncclMin, comms[i], gpus[i].stream));
    NCCL_CHECK(ncclAllReduce(planes_h[i].miny, planes_h[i].miny, 1,
                             ncclFloat, ncclMin, comms[i], gpus[i].stream));
    NCCL_CHECK(ncclAllReduce(planes_h[i].maxx, planes_h[i].maxx, 1,
                             ncclFloat, ncclMax, comms[i], gpus[i].stream));
    NCCL_CHECK(ncclAllReduce(planes_h[i].maxy, planes_h[i].maxy, 1,
                             ncclFloat, ncclMax, comms[i], gpus[i].stream));
  }
  NCCL_CHECK(ncclGroupEnd());

  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(gpus[i].dev));
    CUDA_CHECK(cudaStreamSynchronize(gpus[i].stream));
  }

  // ---- 3. build the global root from the AllReduced bbox -----------------
  float g_minx = 0.f, g_miny = 0.f, g_maxx = 0.f, g_maxy = 0.f;
  CUDA_CHECK(cudaSetDevice(gpus[0].dev));
  CUDA_CHECK(cudaMemcpy(&g_minx, planes_h[0].minx, sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&g_miny, planes_h[0].miny, sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&g_maxx, planes_h[0].maxx, sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&g_maxy, planes_h[0].maxy, sizeof(float),
                        cudaMemcpyDeviceToHost));
  std::printf("\n==== global bounding box ====\n");
  std::printf("min=(%.4f, %.4f)  max=(%.4f, %.4f)\n",
              g_minx, g_miny, g_maxx, g_maxy);

  const float cx = 0.5f * (g_minx + g_maxx);
  const float cy = 0.5f * (g_miny + g_maxy);
  // pad the radius a touch so points right on the global bbox don't fail
  // build_tree's strict `>` test in surprising ways.
  float r = 0.5f * std::fmax(g_maxx - g_minx, g_maxy - g_miny);
  r *= 1.05f;
  root h_root{cx, cy, r};
  std::printf("global root: center=(%.4f, %.4f) radius=%.4f\n", cx, cy, r);

  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(gpus[i].dev));
    CUDA_CHECK(cudaMemcpy(gpus[i].d_root, &h_root, sizeof(root),
                          cudaMemcpyHostToDevice));
  }

  // ---- 4. build_tree (per GPU) -------------------------------------------
  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(gpus[i].dev));
    build_tree<<<1, 64>>>(gpus[i].d_points, number_of_points,
                          gpus[i].d_tree, gpus[i].d_root);
    CUDA_CHECK(cudaGetLastError());
  }
  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(gpus[i].dev));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // ---- 5. summarize_kernel (per GPU) -------------------------------------
  for (int i = 0; i < nDev; ++i) {
    GpuState& g = gpus[i];
    CUDA_CHECK(cudaSetDevice(g.dev));

    std::vector<int>   init_count(g.com_len, -1);
    std::vector<float> init_avg(g.com_len * 2, 0.0f);
    CUDA_CHECK(cudaMemcpy(g.d_count, init_count.data(),
                          sizeof(int) * g.com_len, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_average, init_avg.data(),
                          sizeof(float) * 2 * g.com_len,
                          cudaMemcpyHostToDevice));

    summarize_kernel<<<1, 32>>>(g.d_points, g.d_average, g.d_count, g.d_tree,
                                g.number_of_points,
                                static_cast<int>(g.max_cells));
    CUDA_CHECK(cudaGetLastError());
  }
  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(gpus[i].dev));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // ---- pretty-print: per-GPU tree after summarize ------------------------
  for (int i = 0; i < nDev; ++i) {
    GpuState& g = gpus[i];
    CUDA_CHECK(cudaSetDevice(g.dev));

    std::vector<int>   out_cells(static_cast<size_t>(g.max_cells) * 4u);
    std::vector<int>   out_count(g.com_len);
    std::vector<float> out_avg(g.com_len * 2);
    CUDA_CHECK(cudaMemcpy(out_cells.data(), g.d_cells,
                          sizeof(int) * out_cells.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_count.data(), g.d_count,
                          sizeof(int) * g.com_len, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_avg.data(), g.d_average,
                          sizeof(float) * 2 * g.com_len,
                          cudaMemcpyDeviceToHost));

    const int root_idx = static_cast<int>(g.max_cells) - 1;
    std::printf("\n==== GPU %d tree AFTER summarize (depth cap %d) ====\n",
                g.dev, print_max_depth);
    print_tree(out_cells, host_points[i], out_count, out_avg,
               number_of_points, root_idx, cx, cy, r, print_max_depth);
  }

  // ---- 6. prepare_to_send_levels (per GPU) -------------------------------
  for (int i = 0; i < nDev; ++i) {
    GpuState& g = gpus[i];
    CUDA_CHECK(cudaSetDevice(g.dev));
    CUDA_CHECK(cudaMemsetAsync(g.d_result_average, 0,
                               sizeof(float) * 2 * M, g.stream));
    CUDA_CHECK(cudaMemsetAsync(g.d_result_count, 0,
                               sizeof(int) * M, g.stream));
  }
  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(gpus[i].dev));
    CUDA_CHECK(cudaStreamSynchronize(gpus[i].stream));
  }

  for (int i = 0; i < nDev; ++i) {
    GpuState& g = gpus[i];
    CUDA_CHECK(cudaSetDevice(g.dev));
    prepare_to_send_levels<<<1, number_of_iter>>>(
        g.d_tree, g.d_average, g.d_count,
        g.d_result_average, g.d_result_count,
        number_of_iter, number_of_layers);
    CUDA_CHECK(cudaGetLastError());
  }
  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(gpus[i].dev));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // pre-AllReduce dump of the packed buffer
  std::vector<std::vector<float>> pre_avg(nDev, std::vector<float>(2 * M, 0.f));
  std::vector<std::vector<int>>   pre_cnt(nDev, std::vector<int>(M, 0));
  for (int i = 0; i < nDev; ++i) {
    GpuState& g = gpus[i];
    CUDA_CHECK(cudaSetDevice(g.dev));
    CUDA_CHECK(cudaMemcpy(pre_avg[i].data(), g.d_result_average,
                          sizeof(float) * 2 * M, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pre_cnt[i].data(), g.d_result_count,
                          sizeof(int) * M, cudaMemcpyDeviceToHost));
    std::printf("\n==== GPU %d packed buffer (pre-AllReduce) ====\n", g.dev);
    for (int k = 0; k < M; ++k) {
      std::printf("  slot %2d: count=%7d  avg=(% .4f, % .4f)\n",
                  k, pre_cnt[i][k],
                  pre_avg[i][2 * k], pre_avg[i][2 * k + 1]);
    }
  }

  // ---- 7. ncclAllReduce SUM on the packed buffer -------------------------
  NCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    GpuState& g = gpus[i];
    CUDA_CHECK(cudaSetDevice(g.dev));
    NCCL_CHECK(ncclAllReduce(g.d_result_average, g.d_result_average,
                             2 * M, ncclFloat, ncclSum,
                             comms[i], g.stream));
    NCCL_CHECK(ncclAllReduce(g.d_result_count, g.d_result_count,
                             M, ncclInt32, ncclSum,
                             comms[i], g.stream));
  }
  NCCL_CHECK(ncclGroupEnd());

  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(gpus[i].dev));
    CUDA_CHECK(cudaStreamSynchronize(gpus[i].stream));
  }

  for (int i = 0; i < nDev; ++i) {
    GpuState& g = gpus[i];
    std::vector<float> a(2 * M);
    std::vector<int>   c(M);
    CUDA_CHECK(cudaSetDevice(g.dev));
    CUDA_CHECK(cudaMemcpy(a.data(), g.d_result_average,
                          sizeof(float) * 2 * M, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(c.data(), g.d_result_count,
                          sizeof(int) * M, cudaMemcpyDeviceToHost));
    std::printf("\n==== GPU %d packed buffer (post-AllReduce SUM) ====\n",
                g.dev);
    for (int k = 0; k < M; ++k) {
      std::printf("  slot %2d: count=%7d  avg=(% .4f, % .4f)\n",
                  k, c[k], a[2 * k], a[2 * k + 1]);
    }
  }

  // ---- 8. apply_sumamary_across_nodes (per GPU) --------------------------
  for (int i = 0; i < nDev; ++i) {
    GpuState& g = gpus[i];
    CUDA_CHECK(cudaSetDevice(g.dev));
    apply_sumamary_across_nodes<<<1, number_of_iter>>>(
        g.d_tree, g.d_average, g.d_count,
        g.d_result_average, g.d_result_count,
        number_of_iter, number_of_layers);
    CUDA_CHECK(cudaGetLastError());
  }
  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(gpus[i].dev));
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // ---- pretty-print: per-GPU tree after the cross-node summary -----------
  for (int i = 0; i < nDev; ++i) {
    GpuState& g = gpus[i];
    CUDA_CHECK(cudaSetDevice(g.dev));

    std::vector<int>   out_cells(static_cast<size_t>(g.max_cells) * 4u);
    std::vector<int>   out_count(g.com_len);
    std::vector<float> out_avg(g.com_len * 2);
    CUDA_CHECK(cudaMemcpy(out_cells.data(), g.d_cells,
                          sizeof(int) * out_cells.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_count.data(), g.d_count,
                          sizeof(int) * g.com_len, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out_avg.data(), g.d_average,
                          sizeof(float) * 2 * g.com_len,
                          cudaMemcpyDeviceToHost));

    const int root_idx = static_cast<int>(g.max_cells) - 1;
    std::printf("\n==== GPU %d tree AFTER apply_sumamary_across_nodes "
                "(depth cap %d) ====\n", g.dev, print_max_depth);
    print_tree(out_cells, host_points[i], out_count, out_avg,
               number_of_points, root_idx, cx, cy, r, print_max_depth);
  }

  // ---- cleanup -----------------------------------------------------------
  for (int i = 0; i < nDev; ++i) {
    GpuState& g = gpus[i];
    CUDA_CHECK(cudaSetDevice(g.dev));
    CUDA_CHECK(cudaFree(g.d_points));
    CUDA_CHECK(cudaFree(g.d_cells));
    CUDA_CHECK(cudaFree(g.d_free));
    CUDA_CHECK(cudaFree(g.d_tree));
    CUDA_CHECK(cudaFree(g.d_root));
    CUDA_CHECK(cudaFree(g.d_average));
    CUDA_CHECK(cudaFree(g.d_count));
    CUDA_CHECK(cudaFree(g.d_result_average));
    CUDA_CHECK(cudaFree(g.d_result_count));
    CUDA_CHECK(cudaStreamDestroy(g.stream));
    ncclCommDestroy(comms[i]);
  }

  std::printf("\ndone.\n");
  return 0;
}
