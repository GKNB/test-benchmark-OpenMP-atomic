#include<omp.h>
#include<cstdlib>
#include<iostream>
#include<cstdlib>
#include<cassert>
#include<cmath>

#include<cuda_runtime.h>
#include "cuda_scatter_add.h"

int main()
{
  constexpr int N = 1024*1024*32;
  constexpr int len = 16;

  using dataType = int;

  int *pos = (int*)malloc(sizeof(int) * N);
  dataType* data = (dataType*)malloc(sizeof(dataType) * N);
  for(int i=0; i<N; i++)
  {
    pos[i] = i % len;
    data[i] = 1;
  }

  dataType* res         = (dataType*)malloc(sizeof(dataType) * len);
  dataType* res_omp     = (dataType*)malloc(sizeof(dataType) * len);
  for(int i=0; i<len; i++)
    res[i] = res_omp[i] = 0.0;

  //=====================================Serial running==================================
  for(int i=0; i<N; i++)
    res[pos[i]] += data[i];
  for(int i=0; i<len; i++)
    std::cout << "i = " << i << "\t res[i] = " << res[i] << std::endl;

  //=====================================Openmp CUDA GPU parallel running (use_device_ptr)==================================
#pragma omp target enter data map(to:res_omp[0:len])
#pragma omp target enter data map(to:data[0:N])
#pragma omp target enter data map(to:pos[0:N])
#pragma omp target data use_device_ptr(res_omp,data,pos)
  {
    cuda_scatter_add<dataType>(res_omp, data, pos, N, 128);
//    cudaDeviceSynchronize();  //If I comment it out, result will be incorrect with clang, while the result will always be correct with nvc++ no matter if we comment it out or not    
  }
#pragma omp target exit data map(from:res_omp[0:len])
#pragma omp target exit data map(release:data[0:N])
#pragma omp target exit data map(release:pos[0:N])
//  cudaDeviceSynchronize();
  for(int i=0; i<len; i++)
    std::cout << "i = " << i << "\t res_omp[i] = " << res_omp[i] << std::endl;


  return 0;
}
