#include<cuda.h>
#include<cuda_runtime.h>

#include "cuda_scatter_add.h"

template<typename T>
__global__ void _cuda_atomic_add(T* res, T* data, int* pos, int size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < size)
  {
    atomicAdd(res+pos[i], data[i]);
  }
}

template<typename T>
void cuda_scatter_add(T* res, T* data, int* pos, int size, int nthreads)
{
  _cuda_atomic_add<T><<<(size+nthreads-1)/nthreads, nthreads>>>(res, data, pos, size);
}

template void cuda_scatter_add<int>(int* res, int* data, int* pos, int size, int nthreads);
template void cuda_scatter_add<float>(float* res, float* data, int* pos, int size, int nthreads);
template void cuda_scatter_add<double>(double* res, double* data, int* pos, int size, int nthreads);
