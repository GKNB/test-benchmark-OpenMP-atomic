#include<omp.h>
#include<cstdlib>
#include<iostream>
#include<cstdlib>
#include<cassert>

#include <hip/hip_runtime.h>

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

  using dataType = double;
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
  hipMalloc((void**)&res_d, sizeof(dataType) * len);
  hipMemcpy(res_d, res_h, sizeof(dataType) * len, hipMemcpyHostToDevice);

  dataType* data_d;
  hipMalloc((void**)&data_d, sizeof(dataType) * N);
  hipMemcpy(data_d, data, sizeof(dataType) * N, hipMemcpyHostToDevice);

  int* pos_d;
  hipMalloc((void**)&pos_d, sizeof(int) * N);
  hipMemcpy(pos_d, pos, sizeof(int) * N, hipMemcpyHostToDevice);

  //=====================================Serial running==================================
  double tc = 0.0;
  for(int lp = 0; lp < nlp; lp++)
  {
    tc -= omp_get_wtime();
    for(int i=0; i<N; i++)
      res[pos[i]] += data[i];
    tc += omp_get_wtime();
  }
  std::cout << "Time for serial with N = " << N << " and len = " << len << " is " << tc << std::endl;

  //=====================================HIP running==================================
  tc = 0.0;
  tc -= omp_get_wtime();
  for(int lp = 0; lp < nlp; lp++)
  {
    test_atomic_add<<<N/512,512>>>(res_d, data_d, pos_d, N);
  }
  hipDeviceSynchronize();
  tc += omp_get_wtime();

  hipMemcpy(res_h, res_d, sizeof(dataType) * len, hipMemcpyDeviceToHost);

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
    assert(0 && "Error! Atomic update for HIP fail\n");

  std::cout << "Time for HIP with N = " << N << " and len = " << len << " is " << tc << std::endl;

  return 0;
}
