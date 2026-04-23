// Simple NCCL AllReduce on two GPUs (single process, multi-device).
// Each GPU is filled with its rank value; after AllReduce(SUM), every GPU
// should hold (0 + 1 + ... + nDev - 1) = 1 in every element.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>


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

#define NCCL_CHECK(cmd)                                                        \
  do {                                                                         \
    ncclResult_t r = (cmd);                                                    \
    if (r != ncclSuccess) {                                                    \
      std::fprintf(stderr, "NCCL error %s:%d '%s'\n", __FILE__, __LINE__,      \
                   ncclGetErrorString(r));                                     \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

int main(int argc, char** argv) {
  int nDev = 2;
  int number_of_points = 5;
  int N = number_of_points * 2; 
  int number_of_threads = 256;
  int number_of_blocks = (number_of_points + number_of_threads -1) / number_of_threads;
  int number_of_iterations = 1;



  std::vector<int> devs = {0, 1};
  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < nDev) {
    std::fprintf(stderr, "Need %d GPUs, found %d\n", nDev, deviceCount);
    return EXIT_FAILURE;
  }
  
  // init dummy data
  std::vector<std::vector<float>> host;
  std::vector<float> first = {1, 2, 3, -5, 5, 9, 7, 4, 6, 7};
  std::vector<float> second = {-1, 2, 10, 4, 5, 9, 7, 4, 6, 7};
  host.push_back(first);
  host.push_back(second);

 
  // each node data 
  builder creator;
  std::vector<node*> points_on_gpu;
  std::vector<plane*> planes;
  for(int i = 0; i < nDev; i++){
    CUDA_CHECK(cudaSetDevice(devs[i]));
    points_on_gpu.push_back(creator.create_node(host[i].data(), number_of_points));
    planes.push_back(creator.create_plane(number_of_blocks));
  }

  // nccl init
  std::vector<float*> points(nDev);
  std::vector<cudaStream_t> streams(nDev);
  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(devs[i]));
    CUDA_CHECK(cudaMalloc(&points[i], N * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&streams[i]));

    CUDA_CHECK(cudaMemcpy(points[i], host[i].data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  std::vector<ncclComm_t> comms(nDev);
  NCCL_CHECK(ncclCommInitAll(comms.data(), nDev, devs.data()));


  // start of iterating 
  for(int i = 0; i < number_of_iterations; i++){

  for(int i = 0; i < nDev; i++){
    cudaSetDevice(devs[i]);
    calculate_bounding_box<<<number_of_blocks, number_of_threads>>>(points_on_gpu[i], planes[i]);
  }

  for(int i = 0; i < nDev; i++){
    cudaSetDevice(devs[i]);
    cudaDeviceSynchronize();
  }

  NCCL_CHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {

    CUDA_CHECK(cudaSetDevice(devs[i]));

    NCCL_CHECK(ncclAllReduce(planes[i]->minx, planes[i]->minx, 1, ncclFloat, ncclMin, 
                             comms[i], streams[i]));
    NCCL_CHECK(ncclAllReduce(planes[i]->miny, planes[i]->miny, 1, ncclFloat, ncclMin,
                             comms[i], streams[i]));
    NCCL_CHECK(ncclAllReduce(planes[i]->maxx, planes[i]->maxx, 1, ncclFloat, ncclMax,
                             comms[i], streams[i]));
    NCCL_CHECK(ncclAllReduce(planes[i]->maxy, planes[i]->maxy, 1, ncclFloat, ncclMax,
                             comms[i], streams[i]));
      
  }
  NCCL_CHECK(ncclGroupEnd());

  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(devs[i]));
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }

}



  bool ok = true;
  for (int i = 0; i < nDev; ++i) {
    CUDA_CHECK(cudaSetDevice(devs[i]));
    CUDA_CHECK(cudaFree(points[i]));
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
    ncclCommDestroy(comms[i]);
  }

  std::printf("%s\n", ok ? "OK" : "FAIL");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
