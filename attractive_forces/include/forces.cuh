#pragma once
#include <cuda_runtime.h>

// Placeholder kernel for the attractive-forces sub-project. Computes a SAXPY
// (y = a*x + y) over n elements; swap this out for the real attractive-force
// accumulation once the algorithm lands.
__global__ void saxpy(int n, float a, const float* x, float* y);
