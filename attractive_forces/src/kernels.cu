#include "forces.cuh"

__global__ void saxpy(int n, float a, const float* x, float* y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}
