#ifndef FORWARD_WARP_H
#define FORWARD_WARP_H

// DO NOT re-define GridSamplerInterpolation here.

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N){
  if (CUDA_NUM_THREADS <= 0) return 0;
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#endif