// Single-GPU smoke test for build_tree (no NCCL).
//
// Drops 4 well-separated points -- one per quadrant of [-2,2]^2 -- so the
// expected tree is just the root cell with each body in a distinct child slot
// (no recursive splits needed). After running build_tree, every populated cell
// in the pool is dumped so you can eyeball the result.
//
// Build & run:
//   cmake --build build --target test_build_tree
//   ./build/test_build_tree
//
// NOTE: build_tree (in bounding_box.cuh) currently has compile errors flagged
// in CLAUDE.md (stray `;` after threadIdx.x, root.x vs root->x, undeclared
// `locked`/`cell`/`old_cell`/`dz`, missing `;` after `int patch = -1`,
// `path` vs `patch`, `* 1` vs `+ 1`, atomic ops needing `&` on cells[locked]
// and the `number_of_free_cells` indirection). Fix those before this test
// will link.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "bounding_box.cuh"
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

int main() {
  CUDA_CHECK(cudaSetDevice(0));

  // 4 points, one per quadrant of [-2, 2] x [-2, 2] (root at origin, r=2).
  // Layout is (x0, y0, x1, y1, ...).
  std::vector<float> host_points = {
    -1.0f,  1.0f,   // body 0 -> NW (step 00 = 0)
     1.0f,  1.0f,   // body 1 -> NE (step 01 = 1)
    -1.0f, -1.0f,   // body 2 -> SW (step 10 = 2)
     1.0f, -1.0f,   // body 3 -> SE (step 11 = 3)
  };
  const int number_of_points = static_cast<int>(host_points.size() / 2);

  // tree->cells is flat: children of cell c live at cells[c*4 .. c*4+3].
  // Allocate a comfortable pool so the kernel never trips its "out of cell
  // memory" trap. 8x bodies is the usual Barnes-Hut heuristic.
  const unsigned int max_cells = 8u * static_cast<unsigned int>(number_of_points);
  const size_t cells_array_len = static_cast<size_t>(max_cells) * 4u;

  // Allocate explicitly rather than calling builder.create_tree, because
  // create_tree currently undersizes `cells` and never initializes
  // `number_of_free_cells` (see CLAUDE.md).
  float* d_points = nullptr;
  CUDA_CHECK(cudaMalloc(&d_points, sizeof(float) * host_points.size()));
  CUDA_CHECK(cudaMemcpy(d_points, host_points.data(),
                        sizeof(float) * host_points.size(),
                        cudaMemcpyHostToDevice));

  int*           d_cells   = nullptr;
  bool*          d_is_body = nullptr;
  unsigned int*  d_free    = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cells,   sizeof(int)  * cells_array_len));
  CUDA_CHECK(cudaMalloc(&d_is_body, sizeof(bool) * cells_array_len));
  CUDA_CHECK(cudaMalloc(&d_free,    sizeof(unsigned int)));

  // Initialize: every slot free (-1), no bodies marked.
  // The kernel uses cell index `number_of_cells - 1` as the root and allocates
  // new cells downward via `atomicSub(number_of_free_cells, 1) - 1`. Seeding
  // the counter at `max_cells - 1` makes the next allocation hand out
  // `max_cells - 2` (i.e. one below the reserved root).
  std::vector<int>  init_cells(cells_array_len, -1);
  std::vector<char> init_is_body(cells_array_len, 0);
  unsigned int      init_free = max_cells - 1u;

  CUDA_CHECK(cudaMemcpy(d_cells, init_cells.data(),
                        sizeof(int) * cells_array_len, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_is_body, init_is_body.data(),
                        sizeof(bool) * cells_array_len, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_free, &init_free, sizeof(unsigned int),
                        cudaMemcpyHostToDevice));

  // Pack the tree struct on device.
  tree h_tree{};
  h_tree.number_of_cells      = max_cells;
  h_tree.number_of_free_cells = d_free;
  h_tree.is_body              = d_is_body;
  h_tree.cells                = d_cells;
  tree* d_tree = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tree, sizeof(tree)));
  CUDA_CHECK(cudaMemcpy(d_tree, &h_tree, sizeof(tree), cudaMemcpyHostToDevice));

  // Root spans [-2, 2]^2.
  root h_root{0.0f, 0.0f, 2.0f};
  root* d_root = nullptr;
  CUDA_CHECK(cudaMalloc(&d_root, sizeof(root)));
  CUDA_CHECK(cudaMemcpy(d_root, &h_root, sizeof(root), cudaMemcpyHostToDevice));

  // Single small block; the kernel walks bodies via `i += inc`.
  build_tree<<<1, 32>>>(d_points, number_of_points, d_tree, d_root);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Read back the cells array and the free-pool head, then dump anything
  // non-empty so the layout is easy to inspect.
  std::vector<int> out_cells(cells_array_len);
  CUDA_CHECK(cudaMemcpy(out_cells.data(), d_cells,
                        sizeof(int) * cells_array_len, cudaMemcpyDeviceToHost));
  unsigned int out_free = 0;
  CUDA_CHECK(cudaMemcpy(&out_free, d_free, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));

  std::printf("free pool head: start=%u  end=%u  (cells consumed=%d)\n",
              init_free, out_free,
              static_cast<int>(init_free) - static_cast<int>(out_free));

  for (unsigned int c = 0; c < max_cells; ++c) {
    int v0 = out_cells[c * 4 + 0];
    int v1 = out_cells[c * 4 + 1];
    int v2 = out_cells[c * 4 + 2];
    int v3 = out_cells[c * 4 + 3];
    if (v0 != -1 || v1 != -1 || v2 != -1 || v3 != -1) {
      std::printf("cell %2u: NW=%3d  NE=%3d  SW=%3d  SE=%3d\n",
                  c, v0, v1, v2, v3);
    }
  }

  // Quick pass/fail check on the expected layout: a single populated root cell
  // (index max_cells - 1) holding bodies 0..3, one per quadrant.
  const unsigned int root_idx = max_cells - 1u;
  bool ok =
      out_cells[root_idx * 4 + 0] == 0 &&  // NW
      out_cells[root_idx * 4 + 1] == 1 &&  // NE
      out_cells[root_idx * 4 + 2] == 2 &&  // SW
      out_cells[root_idx * 4 + 3] == 3;    // SE
  std::printf("root layout check: %s\n", ok ? "OK" : "FAIL");

  CUDA_CHECK(cudaFree(d_points));
  CUDA_CHECK(cudaFree(d_cells));
  CUDA_CHECK(cudaFree(d_is_body));
  CUDA_CHECK(cudaFree(d_free));
  CUDA_CHECK(cudaFree(d_tree));
  CUDA_CHECK(cudaFree(d_root));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
