#include<omp.h>
#include<cstdlib>
#include<iostream>
#include<cstdlib>
#include<cassert>

int main()
{
  constexpr int N = 1024 * 1024 * 32;
  constexpr int len = 1024 * 128;
  constexpr int nlp = 10;

  using dataType = float;
  dataType delta = (dataType)2e-2;
  std::cout << "We are testing atomicAdd with type " << typeid(delta).name() << std::endl;
  srand(1234);
  bool do_abort = false;

  int *pos = (int*)malloc(sizeof(int) * N);
  dataType* data = (dataType*)malloc(sizeof(dataType) * N);
  for(int i=0; i<N; i++)
  {
    pos[i] = rand() % len;
    data[i] = rand() / (dataType)RAND_MAX;
  }

  dataType* res       = (dataType*)malloc(sizeof(dataType) * len);
  dataType* res_d     = (dataType*)malloc(sizeof(dataType) * len);
  for(int i=0; i<len; i++)
    res[i] = res_d[i] = 0.0;

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

  //=====================================Openmp GPU parallel running==================================
#pragma omp target enter data map(to:res_d[0:len])
#pragma omp target enter data map(to:data[0:N])
#pragma omp target enter data map(to:pos[0:N])

  //warm-up
#pragma omp target teams distribute parallel for simd
  for(int i=0; i<N; i++)
  {
#pragma omp atomic update
    res_d[pos[i]] += data[i];
  }

  tc = 0.0;
  for(int lp = 0; lp < nlp; lp++)
  {
    tc -= omp_get_wtime();
#pragma omp target teams distribute parallel for simd
    for(int i=0; i<N; i++)
    {
#pragma omp atomic update
      res_d[pos[i]] += data[i];
    }
    tc += omp_get_wtime();
  }
#pragma omp target exit data map(from:res_d[0:len])
#pragma omp target exit data map(release:data[0:N])
#pragma omp target exit data map(release:pos[0:N])

  for(int i=0; i<len; i++)
  {
    if(abs(res_d[i] - res[i]) > delta)
    {
      std::cout << "Error! res_d and res are different at i = " << i << " with res_d[i] = " << res_d[i] << " and res[i] = " << res[i] << std::endl;
      do_abort = true;
    }
  }
  if(do_abort)
    assert(0 && "Error! Atomic update for Openmp GPU fail\n");

  std::cout << "Time for GPU openmp atomic with N = " << N << " and len = " << len << " is " << tc << std::endl;

  return 0;
}
