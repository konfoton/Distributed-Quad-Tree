# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

CMake project (CUDA + NCCL). All build artifacts live in `build/` (gitignored).

```bash
cmake -S . -B build                         # configure
cmake --build build -j                      # build the `main` executable
./build/main                                # run (requires ≥2 CUDA GPUs)
```

Useful overrides:
- `-DCMAKE_CUDA_ARCHITECTURES=80` — narrow GPU arch (defaults to `80;86;90`, i.e. Ampere + Hopper).
- `-DNCCL_ROOT=/path/to/nccl` — point at a non-standard NCCL install. CMake first tries the NCCL CMake config, then falls back to a manual `find_path` / `find_library`.

There is no test suite, lint config, or CI in the repo.

## Architecture

Single-process, multi-GPU implementation of a Barnes–Hut-style quad-tree. All host code lives in `main.cu`; data structures and device kernels live in the two `.cuh` headers and are included directly.

### Device-pointer convention (important)

`builder` (in `objects.cuh`) is the only allocator. Every `create_*` method returns a **device pointer to a struct that itself contains device pointers**:

- `create_node(...)` → `node*` on device. `node.points` is also a device array.
- `create_plane(blocks)` → `plane*` on device. `plane.{minx,miny,maxx,maxy}` are device arrays of length `blocks`.
- `create_tree(...)` → `tree*` on device. `tree.{is_body,cells}` are device arrays.

Consequence: **never dereference these on the host.** To use the inner pointers from host code (e.g. for NCCL calls or `cudaMemcpy`), first copy the outer struct back to host with `cudaMemcpy(&plane_h, plane_d, sizeof(plane), cudaMemcpyDeviceToHost)` and then use `plane_h.minx`. `main.cu` follows this pattern when handing `plane.minx/maxx/...` pointers to `ncclAllReduce`. The note at `main.cu:101` and the recent commit `3489921 ("works! solved problem with dereferencing d on h")` exist because of this exact footgun.

### Per-iteration pipeline (`main.cu`)

For each GPU `i`:
1. `calculate_bounding_box<<<blocks, threads>>>(node[i], plane[i])` — block-level shared-memory reduction over points, then a "last block in" atomic-counter trick (`atomicInc` on `plane->current_number_of_blocks`) lets the last block finalize the per-GPU bbox into `plane->minx[0]`, etc.
2. `cudaDeviceSynchronize()` on each device.
3. Copy each `plane` struct device→host to read the inner device pointers.
4. `ncclGroupStart` / `ncclGroupEnd` around four `ncclAllReduce` calls (`ncclMin` for min coords, `ncclMax` for max coords) so every GPU ends up with the global bounding box at `planes_h[i].minx[0]` etc.
5. Stream sync, then `cudaMemcpy` element 0 back to host for printing.

### Quad-tree encoding (`objects.cuh`)

Quadrant indexing used throughout `build_tree`:
- Bit 0 = "x > center" (east), bit 1 = "y > center" (north). So `0 = 00 = SW`, `1 = 01 = SE`, `2 = 10 = NW`, `3 = 11 = NE`.
- For cell index `i`, children are `cells[i*4 + 0..3]` in the order SW, SE, NW, NE.
- Cell sentinel values in `tree.cells`: `-1` = free, `-2` = locked (during insertion).
- A non-sentinel value is a body index when `is_body[i]` is true, otherwise a child cell index.

Insertion (`build_tree` in `bounding_box.cuh`) is the standard lock-via-`atomicCAS` Barnes–Hut warp-cooperative scheme: descend until you hit `-1` (claim with CAS) or another body (CAS to `-2`, then split-and-relink in the `do/while` loop, allocating new cells from a free pool via `atomicSub` on `tree->number_of_free_cells`). `MAXDEPTH` (32) traps execution if two points are too close.

### State of the code

`main.cu` and `calculate_bounding_box` work end-to-end. `build_tree`, `clear_kernel`, `summarize_kernel`, and `traverse_tree` in `bounding_box.cuh` are **work-in-progress** and contain known issues — typos like `int i = threadIdx.x; + blockIdx.x * blockDim.x;` (stray `;`), use of `dz` in 2-D code, missing declarations (`locked`, `old_cell`, `cell`), `path` vs `patch`, missing semicolons, and `root.x` used where `root->x` is needed. Don't assume these compile; treat them as scaffolding for the algorithm sketched in the comments (force accumulation with the `r_cell/d < theta` Barnes–Hut criterion).
