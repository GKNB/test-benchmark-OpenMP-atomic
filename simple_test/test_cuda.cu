#include<omp.h>
#include<cstdlib>
#include<iostream>
#include<cstdlib>
#include<cassert>

#include<cuda.h>

template<typename T>
__global__ void test_atomic_add(T* res, T* data, int* pos, int size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < size)
  {
    atomicAdd(res+pos[i], data[i]);
  }
}

int main()
{
  constexpr int N = 1024 * 1024 * 32;
  constexpr int len = 1024 * 128;
  constexpr int nlp = 10;
  constexpr int num_threads_per_block = 256;

  using dataType = float;
  dataType delta = (dataType)2e-2;
  std::cout << "We are testing atomicAdd with type " << typeid(delta).name() << std::endl;
  srand(1234);

  int *pos = (int*)malloc(sizeof(int) * N);
  dataType* data = (dataType*)malloc(sizeof(dataType) * N);
  for(int i=0; i<N; i++)
  {
    pos[i] = rand() % len;
    data[i] = rand() / (dataType)RAND_MAX;
  }

  dataType* res   = (dataType*)malloc(sizeof(dataType) * len);
  dataType* res_h = (dataType*)malloc(sizeof(dataType) * len);
  for(int i=0; i<len; i++)
    res[i] = res_h[i] = 0.0;

  dataType* res_d;
  cudaMalloc((void**)&res_d, sizeof(dataType) * len);
  cudaMemcpy(res_d, res_h, sizeof(dataType) * len, cudaMemcpyHostToDevice);

  dataType* data_d;
  cudaMalloc((void**)&data_d, sizeof(dataType) * N);
  cudaMemcpy(data_d, data, sizeof(dataType) * N, cudaMemcpyHostToDevice);

  int* pos_d;
  cudaMalloc((void**)&pos_d, sizeof(int) * N);
  cudaMemcpy(pos_d, pos, sizeof(int) * N, cudaMemcpyHostToDevice);

  //=====================================Serial running==================================
  //
  //warm-up
  for(int i=0; i<N; i++)
    res[pos[i]] += data[i];

  double tc = 0.0;
  for(int lp = 0; lp < nlp; lp++)
  {
    tc -= omp_get_wtime();
    for(int i=0; i<N; i++)
      res[pos[i]] += data[i];
    tc += omp_get_wtime();
  }
  std::cout << "Time for serial with N = " << N << " and len = " << len << " is " << tc << std::endl;

  //=====================================CUDA running==================================
  //
  //warm-up

  test_atomic_add<<<N/num_threads_per_block,num_threads_per_block>>>(res_d, data_d, pos_d, N);
  cudaDeviceSynchronize();

  tc = 0.0;
  tc -= omp_get_wtime();
  for(int lp = 0; lp < nlp; lp++)
  {
    test_atomic_add<<<N/num_threads_per_block,num_threads_per_block>>>(res_d, data_d, pos_d, N);
  }
  cudaDeviceSynchronize();
  tc += omp_get_wtime();

  cudaMemcpy(res_h, res_d, sizeof(dataType) * len, cudaMemcpyDeviceToHost);

  bool do_abort = false;
  for(int i=0; i<len; i++)
  {
    if(abs(res_h[i] - res[i]) > delta)
    {
      std::cout << "Error! res_h and res are different at i = " << i << " with res_h[i] = " << res_h[i] << " and res[i] = " << res[i] << std::endl;
      do_abort = true;
    }
  }
  if(do_abort)
    assert(0 && "Error! Atomic update for CUDA fail\n");

  std::cout << "Time for CUDA with N = " << N << " and len = " << len << " is " << tc << std::endl;

  return 0;
}
