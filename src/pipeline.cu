// Full multi-GPU Barnes-Hut pipeline for profiling. No prints.
//
// 4 GPUs. Each GPU owns one quadrant of the global [-1, 1]^2 space:
//   GPU 0 -> SW   ( x in [-1, 0), y in [-1, 0) )
//   GPU 1 -> SE   ( x in [ 0, 1), y in [-1, 0) )
//   GPU 2 -> NW   ( x in [-1, 0), y in [ 0, 1) )
//   GPU 3 -> NE   ( x in [ 0, 1), y in [ 0, 1) )
//
// Per-iteration pipeline (single iteration here; loop the for-body to bench
// repeated steps):
//   1. calculate_bounding_box        per GPU
//   2. ncclAllReduce  min/max bbox   -> global root identical on every GPU
//   3. build_tree                    per GPU
//   4. clear_kernel_two + summarize_kernel
//   5. prepare_to_send_levels + ncclAllReduce(SUM) + apply_sumamary_across_nodes
//   6. ClearKernelthree + SortNodes
//   7. traverse_tree                 per GPU (writes per-body gradient)
//
// Build & run:
//   cmake --build build --target pipeline
//   ./build/pipeline                  # requires 4 CUDA GPUs with NCCL

#include <cuda_runtime.h>
#include <nccl.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

// ---- knobs ---------------------------------------------------------------
constexpr int   kNumDev          = 4;
constexpr int   kPointsPerGpu    = 1'000'000;
constexpr int   kCellsHeuristic  = 8;        // max_cells = X * N
constexpr int   kIterations      = 10;        // bench loop count

// share root + 2 levels = 1 + 4 + 16 = 21 cells across nodes
constexpr int   kSendLayers      = 2;
constexpr int   kSendIter        = 1 << (2 * kSendLayers);  // 16

// kernel launch shapes
constexpr int   kThreads         = 256;          // most kernels
constexpr int   kBuildBlocks     = 64;          // contention on atomicCAS
constexpr int   kBuildThreads    = 256;
constexpr int   kSummarizeBlocks = 64;
constexpr int   kSummarizeThreads= 256;          // must be <= max_threads (=256)
constexpr int   kSortBlocks      = 64;
constexpr int   kSortThreads     = 256;
constexpr int   kTraverseThreads = 256;          // multiple of WARPSIZE

// Barnes-Hut tolerance
constexpr float kTheta           = 0.5f;
constexpr float kEps             = 1e-3f;

int compute_M(int layers) {
  int M = 0, term = 1;
  for (int j = 0; j <= layers; ++j) { M += term; term *= 4; }
  return M;
}

// Deterministic LCG -> [0, 1).
float lcg(std::uint32_t& state) {
  state = state * 1664525u + 1013904223u;
  return (state >> 8) * (1.0f / 16777216.0f);
}

// Generate `n` points biased to a single quadrant. (qx, qy) in {-1, +1}
// picks the quadrant origin: (-1,-1) SW, (+1,-1) SE, (-1,+1) NW, (+1,+1) NE.
// Points land in [qx*0..qx*1) x [qy*0..qy*1) (size 1 quadrant), keeping
// each GPU's cloud spatially local.
void fill_quadrant(std::vector<float>& out, int n,
                   float qx, float qy, std::uint32_t seed) {
  out.resize(n * 2);
  std::uint32_t s = seed;
  const float ox = (qx < 0.f) ? -1.0f : 0.0f;
  const float oy = (qy < 0.f) ? -1.0f : 0.0f;
  for (int i = 0; i < n; ++i) {
    out[i * 2 + 0] = ox + lcg(s);   // [ox, ox + 1)
    out[i * 2 + 1] = oy + lcg(s);
  }
}

struct GpuState {
  int dev;

  node*  d_node;
  plane* d_plane;
  tree*  d_tree;
  root*  d_root;

  float*        d_points;
  int*          d_cells;
  unsigned int* d_free;
  float*        d_average;
  int*          d_count_of_points;
  int*          d_count;
  int*          d_sorted;
  float*        d_gradient;

  float* d_result_average;   // size 2 * M
  int*   d_result_count;     // size M

  cudaStream_t stream;

  int          number_of_points;
  unsigned int max_cells;
  size_t       com_len;
  int          bbox_blocks;
};

}  // namespace

int main() {
  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < kNumDev) {
    std::fprintf(stderr, "need %d GPUs, found %d\n", kNumDev, deviceCount);
    return EXIT_FAILURE;
  }

  std::vector<int> devs(kNumDev);
  for (int i = 0; i < kNumDev; ++i) devs[i] = i;

  // Quadrant assignment per GPU: SW, SE, NW, NE.
  const float quad_x[kNumDev] = {-1.f, +1.f, -1.f, +1.f};
  const float quad_y[kNumDev] = {-1.f, -1.f, +1.f, +1.f};
  const std::uint32_t seeds[kNumDev] = {
      0xC0FFE001u, 0xC0FFE002u, 0xC0FFE003u, 0xC0FFE004u};

  // ---- 0. host-side per-quadrant point clouds ----------------------------
  std::vector<std::vector<float>> host_points(kNumDev);
  for (int i = 0; i < kNumDev; ++i) {
    fill_quadrant(host_points[i], kPointsPerGpu,
                  quad_x[i], quad_y[i], seeds[i]);
  }

  const int M = compute_M(kSendLayers);

  // ---- 1. NCCL init + per-GPU buffer allocation --------------------------
  std::vector<GpuState> gpus(kNumDev);
  std::vector<ncclComm_t> comms(kNumDev);
  NCCL_CHECK(ncclCommInitAll(comms.data(), kNumDev, devs.data()));

  builder creator;

  for (int i = 0; i < kNumDev; ++i) {
    GpuState& g = gpus[i];
    g.dev               = devs[i];
    g.number_of_points  = kPointsPerGpu;
    g.max_cells         = kCellsHeuristic *
                          static_cast<unsigned int>(kPointsPerGpu);
    g.com_len           = static_cast<size_t>(kPointsPerGpu) + g.max_cells;
    g.bbox_blocks       = (kPointsPerGpu + kThreads - 1) / kThreads;
    if (g.bbox_blocks > 1024) g.bbox_blocks = 1024;

    CUDA_CHECK(cudaSetDevice(g.dev));
    CUDA_CHECK(cudaStreamCreate(&g.stream));

    g.d_node  = creator.create_node(host_points[i].data(), kPointsPerGpu);
    g.d_plane = creator.create_plane(g.bbox_blocks);

    // Re-use the device points array allocated inside d_node so we don't
    // pay for a duplicate copy.
    node h_node;
    CUDA_CHECK(cudaMemcpy(&h_node, g.d_node, sizeof(node),
                          cudaMemcpyDeviceToHost));
    g.d_points = h_node.points;

    const size_t cells_array_len =
        static_cast<size_t>(g.max_cells) * 4u;
    CUDA_CHECK(cudaMalloc(&g.d_cells, sizeof(int) * cells_array_len));
    CUDA_CHECK(cudaMalloc(&g.d_free,  sizeof(unsigned int)));
    // memset to 0xFF gives every int -1 in one shot (cheaper than a HtoD
    // copy of an N-sized vector).
    CUDA_CHECK(cudaMemset(g.d_cells, 0xFF, sizeof(int) * cells_array_len));
    unsigned int init_free = g.max_cells - 1u;
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

    CUDA_CHECK(cudaMalloc(&g.d_average,
                          sizeof(float) * 2 * g.com_len));
    CUDA_CHECK(cudaMalloc(&g.d_count_of_points,
                          sizeof(int)   * g.com_len));
    CUDA_CHECK(cudaMalloc(&g.d_count,  sizeof(int)   * g.com_len));
    CUDA_CHECK(cudaMalloc(&g.d_sorted, sizeof(int)   * kPointsPerGpu));
    CUDA_CHECK(cudaMalloc(&g.d_gradient,
                          sizeof(float) * 2 * kPointsPerGpu));

    CUDA_CHECK(cudaMalloc(&g.d_result_average, sizeof(float) * 2 * M));
    CUDA_CHECK(cudaMalloc(&g.d_result_count,   sizeof(int)   * M));
  }
  // -------------------------------------------------------------------------
  // Iteration loop. Wrap stages 1..7 to repeat-time benchmark a steady-state
  // step once buffers are warm. Increase kIterations to deepen the trace.
  // -------------------------------------------------------------------------
  for (int iter = 0; iter < kIterations; ++iter) {

    // ---- 1. calculate_bounding_box (per GPU) -----------------------------
    for (int i = 0; i < kNumDev; ++i) {
      CUDA_CHECK(cudaSetDevice(gpus[i].dev));
      // reset the per-plane block counter so the "last block in" reduction
      // works across iterations.
      CUDA_CHECK(cudaMemsetAsync(
          &gpus[i].d_plane->current_number_of_blocks, 0,
          sizeof(unsigned int), gpus[i].stream));
      calculate_bounding_box<<<gpus[i].bbox_blocks, kThreads, 0,
                               gpus[i].stream>>>(
          gpus[i].d_node, gpus[i].d_plane);
    }
    for (int i = 0; i < kNumDev; ++i) {
      CUDA_CHECK(cudaSetDevice(gpus[i].dev));
      CUDA_CHECK(cudaStreamSynchronize(gpus[i].stream));
    }

    // ---- 2. ncclAllReduce on bboxes --------------------------------------
    std::vector<plane> planes_h(kNumDev);
    for (int i = 0; i < kNumDev; ++i) {
      CUDA_CHECK(cudaSetDevice(gpus[i].dev));
      CUDA_CHECK(cudaMemcpy(&planes_h[i], gpus[i].d_plane, sizeof(plane),
                            cudaMemcpyDeviceToHost));
    }

    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < kNumDev; ++i) {
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
    for (int i = 0; i < kNumDev; ++i) {
      CUDA_CHECK(cudaSetDevice(gpus[i].dev));
      CUDA_CHECK(cudaStreamSynchronize(gpus[i].stream));
    }

    // ---- 3. build_tree (per GPU) -----------------------------------------
    // Pull the global bbox off GPU 0 and build a single root, then broadcast
    // it to every GPU's d_root.
    float g_minx, g_miny, g_maxx, g_maxy;
    CUDA_CHECK(cudaSetDevice(gpus[0].dev));
    CUDA_CHECK(cudaMemcpy(&g_minx, planes_h[0].minx, sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&g_miny, planes_h[0].miny, sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&g_maxx, planes_h[0].maxx, sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&g_maxy, planes_h[0].maxy, sizeof(float),
                          cudaMemcpyDeviceToHost));
    const float cx = 0.5f * (g_minx + g_maxx);
    const float cy = 0.5f * (g_miny + g_maxy);
    float r = 0.5f * std::fmax(g_maxx - g_minx, g_maxy - g_miny);
    r *= 1.05f;   // pad so points exactly on the bbox don't trip strict `>`
    root h_root{cx, cy, r};

    for (int i = 0; i < kNumDev; ++i) {
      CUDA_CHECK(cudaSetDevice(gpus[i].dev));
      CUDA_CHECK(cudaMemcpyAsync(gpus[i].d_root, &h_root, sizeof(root),
                                 cudaMemcpyHostToDevice, gpus[i].stream));
    }
    for (int i = 0; i < kNumDev; ++i) {
      CUDA_CHECK(cudaSetDevice(gpus[i].dev));
      build_tree<<<kBuildBlocks, kBuildThreads, 0, gpus[i].stream>>>(
          gpus[i].d_points, gpus[i].number_of_points,
          gpus[i].d_tree, gpus[i].d_root);
    }
    for (int i = 0; i < kNumDev; ++i) {
      CUDA_CHECK(cudaSetDevice(gpus[i].dev));
      CUDA_CHECK(cudaStreamSynchronize(gpus[i].stream));
    }
    // ---- 4. clear_kernel_two + summarize_kernel --------------------------
    for (int i = 0; i < kNumDev; ++i) {
      GpuState& g = gpus[i];
      CUDA_CHECK(cudaSetDevice(g.dev));
      const int clear_blocks =
          (static_cast<int>(g.com_len) + kThreads - 1) / kThreads;
      clear_kernel_two<<<clear_blocks, kThreads, 0, g.stream>>>(
          g.d_average, g.d_count_of_points, static_cast<int>(g.com_len));
      summarize_kernel<<<kSummarizeBlocks, kSummarizeThreads, 0, g.stream>>>(
          g.d_points, g.d_average, g.d_count_of_points, g.d_tree,
          g.number_of_points, static_cast<int>(g.max_cells));
    }
    for (int i = 0; i < kNumDev; ++i) {
      CUDA_CHECK(cudaSetDevice(gpus[i].dev));
      CUDA_CHECK(cudaStreamSynchronize(gpus[i].stream));
    }
    // ---- 5. prepare_to_send_levels + AllReduce + apply_summary -----------
    for (int i = 0; i < kNumDev; ++i) {
      GpuState& g = gpus[i];
      CUDA_CHECK(cudaSetDevice(g.dev));
      CUDA_CHECK(cudaMemsetAsync(g.d_result_average, 0,
                                 sizeof(float) * 2 * M, g.stream));
      CUDA_CHECK(cudaMemsetAsync(g.d_result_count, 0,
                                 sizeof(int) * M, g.stream));
      prepare_to_send_levels<<<1, kSendIter, 0, g.stream>>>(
          g.d_tree, g.d_average, g.d_count_of_points,
          g.d_result_average, g.d_result_count,
          kSendIter, kSendLayers, g.number_of_points);
    }
    for (int i = 0; i < kNumDev; ++i) {
      CUDA_CHECK(cudaSetDevice(gpus[i].dev));
      CUDA_CHECK(cudaStreamSynchronize(gpus[i].stream));
    }


    NCCL_CHECK(ncclGroupStart());
    for (int i = 0; i < kNumDev; ++i) {
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
    for (int i = 0; i < kNumDev; ++i) {
      CUDA_CHECK(cudaSetDevice(gpus[i].dev));
      CUDA_CHECK(cudaStreamSynchronize(gpus[i].stream));
    }

    for (int i = 0; i < kNumDev; ++i) {
      GpuState& g = gpus[i];
      CUDA_CHECK(cudaSetDevice(g.dev));
      apply_sumamary_across_nodes<<<1, kSendIter, 0, g.stream>>>(
          g.d_tree, g.d_average, g.d_count_of_points,
          g.d_result_average, g.d_result_count,
          kSendIter, kSendLayers, g.number_of_points);
    }
    for (int i = 0; i < kNumDev; ++i) {
      CUDA_CHECK(cudaSetDevice(gpus[i].dev));
      CUDA_CHECK(cudaStreamSynchronize(gpus[i].stream));
    }
    // ---- 6. ClearKernelthree + SortNodes ---------------------------------
    for (int i = 0; i < kNumDev; ++i) {
      GpuState& g = gpus[i];
      CUDA_CHECK(cudaSetDevice(g.dev));
      ClearKernelthree<<<kSortBlocks, kSortThreads, 0, g.stream>>>(
          g.d_count, g.d_tree);
      SortNodes<<<kSortBlocks, kSortThreads, 0, g.stream>>>(
          g.d_count, g.d_sorted, g.d_points, g.d_count_of_points,
          g.d_tree, g.number_of_points, static_cast<int>(g.max_cells));
    }
    for (int i = 0; i < kNumDev; ++i) {
      CUDA_CHECK(cudaSetDevice(gpus[i].dev));
      CUDA_CHECK(cudaStreamSynchronize(gpus[i].stream));
    }
    
    // ---- 7. traverse_tree (per GPU) --------------------------------------
    const float itolsqd = 1.0f / (kTheta * kTheta);
    const int   trav_blocks =
        (kPointsPerGpu + kTraverseThreads - 1) / kTraverseThreads;
    for (int i = kNumDev - 1; i >= 0; --i) {
      GpuState& g = gpus[i];
      CUDA_CHECK(cudaSetDevice(g.dev));
      traverse_tree<<<trav_blocks, kTraverseThreads, 0, g.stream>>>(
          g.d_tree, g.d_root, itolsqd, kEps,
          g.d_sorted, g.d_average, g.d_count_of_points,
          static_cast<int>(g.max_cells), g.number_of_points,
          g.d_points, g.d_gradient);
    }
    for (int i = kNumDev - 1; i >= 0; --i) {
      CUDA_CHECK(cudaSetDevice(gpus[i].dev));
      CUDA_CHECK(cudaStreamSynchronize(gpus[i].stream));
    }

    // ---- end of one iteration. Reset cells/free counter for the next one
    // so build_tree starts from a clean slate. (Skipped on the last iter.)
    if (iter + 1 < kIterations) {
      for (int i = 0; i < kNumDev; ++i) {
        GpuState& g = gpus[i];
        CUDA_CHECK(cudaSetDevice(g.dev));
        const size_t cells_array_len =
            static_cast<size_t>(g.max_cells) * 4u;
        CUDA_CHECK(cudaMemsetAsync(g.d_cells, 0xFF,
                                   sizeof(int) * cells_array_len, g.stream));
        unsigned int init_free = g.max_cells - 1u;
        CUDA_CHECK(cudaMemcpyAsync(g.d_free, &init_free,
                                   sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, g.stream));
      }
      for (int i = 0; i < kNumDev; ++i) {
        CUDA_CHECK(cudaSetDevice(gpus[i].dev));
        CUDA_CHECK(cudaStreamSynchronize(gpus[i].stream));
      }
    }
  }

  // ---- cleanup -----------------------------------------------------------
  for (int i = 0; i < kNumDev; ++i) {
    GpuState& g = gpus[i];
    CUDA_CHECK(cudaSetDevice(g.dev));
    CUDA_CHECK(cudaFree(g.d_cells));
    CUDA_CHECK(cudaFree(g.d_free));
    CUDA_CHECK(cudaFree(g.d_tree));
    CUDA_CHECK(cudaFree(g.d_root));
    CUDA_CHECK(cudaFree(g.d_average));
    CUDA_CHECK(cudaFree(g.d_count_of_points));
    CUDA_CHECK(cudaFree(g.d_count));
    CUDA_CHECK(cudaFree(g.d_sorted));
    CUDA_CHECK(cudaFree(g.d_gradient));
    CUDA_CHECK(cudaFree(g.d_result_average));
    CUDA_CHECK(cudaFree(g.d_result_count));
    CUDA_CHECK(cudaStreamDestroy(g.stream));
    ncclCommDestroy(comms[i]);
  }
  return 0;
}
