#ifndef _CUDA_SCATTER_ADD_H_
#define _CUDA_SCATTER_ADD_H_

template<typename T>
void cuda_scatter_add(T* res, T* data, int* pos, int size, int nthreads);

#endif
