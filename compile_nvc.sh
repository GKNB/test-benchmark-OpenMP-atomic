nvcc test_cuda.cu -Xcompiler -fopenmp -lgomp -arch=sm_70 -o run_cuda_nvcc
nvc++ test_openmp.C -mp=gpu -cuda -Minfo=mp -o run_openmp
