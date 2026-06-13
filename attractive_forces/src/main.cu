// Dummy single-GPU CUDA program for the attractive_forces sub-project.
// Runs a SAXPY (y = a*x + y) on the device and verifies the result on the
// host, mostly to prove the monorepo build wires this target up correctly.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "forces.cuh"

#define CUDA_CHECK(cmd)                                                   \
  do {                                                                    \
    cudaError_t e = (cmd);                                                \
    if (e != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                                \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

int main() {
  const int n = 1 << 20;  // 1M elements
  const float a = 2.0f;

  std::vector<float> h_x(n, 1.0f);
  std::vector<float> h_y(n, 2.0f);

  float *d_x = nullptr, *d_y = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  saxpy<<<blocks, threads>>>(n, a, d_x, d_y);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));

  // Expected: a*1 + 2 = 4 everywhere.
  float max_err = 0.0f;
  for (int i = 0; i < n; ++i) {
    max_err = std::fmax(max_err, std::fabs(h_y[i] - 4.0f));
  }
  std::printf("attractive_forces dummy: max error = %f\n", max_err);

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  return max_err == 0.0f ? EXIT_SUCCESS : EXIT_FAILURE;
}
