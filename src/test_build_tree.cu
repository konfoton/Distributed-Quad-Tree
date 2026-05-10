// Single-GPU smoke test for build_tree (no NCCL).
//
// Two test cases:
//   1. Level-1 layout: 4 well-separated points, one per root quadrant. The
//      expected tree is just the root cell with each body in a distinct slot
//      (no recursive splits needed).
//   2. Level-2 layout: 5 points, two of which share the root NE quadrant and
//      therefore force build_tree to allocate a child cell from the free pool.
//
// Quadrant convention enforced by build_tree:
//   bit 0 of step is set when x > center.x  (east)
//   bit 1 of step is set when y > center.y  (north)
//   => step 0=SW, 1=SE, 2=NW, 3=NE   (children live at cells[i*4 + step]).
//
// After each kernel run we walk the cells array recursively and pretty-print
// the tree so the layout is easy to eyeball.
//
// Build & run:
//   cmake --build build --target test_build_tree
//   ./build/test_build_tree
//
// NOTE: build_tree (in bounding_box.cuh) still has known issues flagged in
// CLAUDE.md (subdivide path uses wrong index arithmetic in a couple of
// places). The level-1 test does not exercise that path; the level-2 test
// does, so its tree may be malformed. The pretty-printer makes that visible.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <string>
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

namespace {

const char* const kQuadrantName[4] = {"SW", "SE", "NW", "NE"};
// Offsets of each child's center relative to its parent's center, in units of
// child radius (= 0.5 * parent radius). Indexed by step (0=SW..3=NE).
const float kDx[4] = {-1.0f, +1.0f, -1.0f, +1.0f};
const float kDy[4] = {-1.0f, -1.0f, +1.0f, +1.0f};

void print_subtree(const std::vector<int>& cells,
                   const std::vector<float>& points, int number_of_points,
                   int cell_idx, float cx, float cy, float r,
                   const std::string& prefix) {
  const float child_r = r * 0.5f;
  for (int s = 0; s < 4; ++s) {
    const bool last = (s == 3);
    const std::string branch = prefix + (last ? "`-- " : "|-- ");
    const std::string sub_prefix = prefix + (last ? "    " : "|   ");

    const int v = cells[cell_idx * 4 + s];
    const float ccx = cx + kDx[s] * child_r;
    const float ccy = cy + kDy[s] * child_r;

    if (v == -1) {
      std::printf("%s%s (step %d): empty\n", branch.c_str(), kQuadrantName[s],
                  s);
    } else if (v == -2) {
      std::printf("%s%s (step %d): LOCKED (-2)\n", branch.c_str(),
                  kQuadrantName[s], s);
    } else if (v < number_of_points) {
      std::printf("%s%s (step %d): body %d @ (%.3f, %.3f)\n", branch.c_str(),
                  kQuadrantName[s], s, v, points[v * 2], points[v * 2 + 1]);
    } else {
      std::printf("%s%s (step %d): cell %d  center=(%.3f, %.3f) r=%.3f\n",
                  branch.c_str(), kQuadrantName[s], s, v, ccx, ccy, child_r);
      print_subtree(cells, points, number_of_points, v, ccx, ccy, child_r,
                    sub_prefix);
    }
  }
}

void print_tree(const std::vector<int>& cells, const std::vector<float>& points,
                int number_of_points, int root_idx, float cx, float cy,
                float r) {
  std::printf("[root cell %d] center=(%.3f, %.3f) r=%.3f\n", root_idx, cx, cy,
              r);
  print_subtree(cells, points, number_of_points, root_idx, cx, cy, r,
                std::string());
}

// Walk the tree and collect every body index reachable from the root. Used as
// a generic correctness check: every body should appear exactly once.
void collect_bodies(const std::vector<int>& cells, int number_of_points,
                    int cell_idx, std::vector<int>& found) {
  for (int s = 0; s < 4; ++s) {
    const int v = cells[cell_idx * 4 + s];
    if (v < 0) continue;  // -1 (empty) or -2 (locked)
    if (v < number_of_points) {
      found.push_back(v);
    } else {
      collect_bodies(cells, number_of_points, v, found);
    }
  }
}

bool run_case(const char* label, const std::vector<float>& host_points,
              float root_x, float root_y, float root_r) {
  std::printf("\n==== %s ====\n", label);

  const int number_of_points = static_cast<int>(host_points.size() / 2);

  // 8x bodies is the usual Barnes-Hut heuristic; bump the floor so even tiny
  // tests have room for a couple of subdivisions.
  unsigned int max_cells = 8u * static_cast<unsigned int>(number_of_points);
  if (max_cells < 16u) max_cells = 16u;
  const size_t cells_array_len = static_cast<size_t>(max_cells) * 4u;

  float* d_points = nullptr;
  CUDA_CHECK(cudaMalloc(&d_points, sizeof(float) * host_points.size()));
  CUDA_CHECK(cudaMemcpy(d_points, host_points.data(),
                        sizeof(float) * host_points.size(),
                        cudaMemcpyHostToDevice));

  int* d_cells = nullptr;
  unsigned int* d_free = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cells, sizeof(int) * cells_array_len));
  CUDA_CHECK(cudaMalloc(&d_free, sizeof(unsigned int)));

  // Every slot empty (-1) and is_body cleared. The root lives at cell index
  // `max_cells - 1`; new cells are allocated downward via
  // `atomicSub(number_of_free_cells, 1) - 1`. Seeding the counter at
  // `max_cells - 1` makes the next allocation hand out `max_cells - 2`.
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

  build_tree<<<1, 32>>>(d_points, number_of_points, d_tree, d_root);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int> out_cells(cells_array_len);
  CUDA_CHECK(cudaMemcpy(out_cells.data(), d_cells,
                        sizeof(int) * cells_array_len, cudaMemcpyDeviceToHost));
  unsigned int out_free = 0;
  CUDA_CHECK(cudaMemcpy(&out_free, d_free, sizeof(unsigned int),
                        cudaMemcpyDeviceToHost));

  const int cells_consumed =
      static_cast<int>(init_free) - static_cast<int>(out_free);
  std::printf("free pool head: start=%u  end=%u  (cells consumed=%d)\n",
              init_free, out_free, cells_consumed);

  const int root_idx = static_cast<int>(max_cells) - 1;
  print_tree(out_cells, host_points, number_of_points, root_idx, root_x, root_y,
             root_r);

  // Reachability check: walking from the root must find every body once.
  std::vector<int> found;
  collect_bodies(out_cells, number_of_points, root_idx, found);
  std::vector<int> seen(number_of_points, 0);
  bool reach_ok = (static_cast<int>(found.size()) == number_of_points);
  for (int b : found) {
    if (b < 0 || b >= number_of_points) {
      reach_ok = false;
      break;
    }
    if (++seen[b] > 1) {
      reach_ok = false;
    }
  }
  std::printf("reachability check: each body visited exactly once -> %s\n",
              reach_ok ? "OK" : "FAIL");

  CUDA_CHECK(cudaFree(d_points));
  CUDA_CHECK(cudaFree(d_cells));
  CUDA_CHECK(cudaFree(d_free));
  CUDA_CHECK(cudaFree(d_tree));
  CUDA_CHECK(cudaFree(d_root));

  return reach_ok;
}

}  // namespace

int main() {
  CUDA_CHECK(cudaSetDevice(0));

  // -- Level 1: one body per root quadrant, root spans [-2, 2]^2. -----------
  // Indexing reminder: step = (y>0 ? 2 : 0) | (x>0 ? 1 : 0).
  //   (-1,-1) -> SW (step 0)
  //   ( 1,-1) -> SE (step 1)
  //   (-1, 1) -> NW (step 2)
  //   ( 1, 1) -> NE (step 3)
  std::vector<float> level1 = {
      -1.0f, -1.0f,  // body 0 -> SW
      1.0f,  -1.0f,  // body 1 -> SE
      -1.0f, 1.0f,   // body 2 -> NW
      1.0f,  1.0f,   // body 3 -> NE
  };
  const bool ok1 =
      run_case("level-1: 4 bodies, no subdivision", level1, 0.0f, 0.0f, 2.0f);

  // -- Level 2: bodies 3 and 4 share the root NE quadrant. ------------------
  // Root NE child has center (1,1) radius 1 (spans [0,2]^2):
  //   ( 1.5,  0.5) -> SE of NE  (sub-step 1)
  //   ( 0.5,  1.5) -> NW of NE  (sub-step 2)
  // The other three bodies sit alone in distinct root quadrants.
  std::vector<float> level2 = {
      -1.5f, -1.5f,  // body 0 -> SW of root
      1.5f,  -1.5f,  // body 1 -> SE of root
      -1.5f, 1.5f,   // body 2 -> NW of root
      1.5f,  0.5f,   // body 3 -> NE of root, then SE of NE
      0.5f,  1.5f,   // body 4 -> NE of root, then NW of NE
  };
  const bool ok2 = run_case("level-2: 5 bodies, NE forces a subdivision",
                            level2, 0.0f, 0.0f, 2.0f);

  std::printf("\n==== summary ====\n");
  std::printf("level-1: %s\n", ok1 ? "OK" : "FAIL");
  std::printf("level-2: %s\n", ok2 ? "OK" : "FAIL");

  return (ok1 && ok2) ? EXIT_SUCCESS : EXIT_FAILURE;
}
