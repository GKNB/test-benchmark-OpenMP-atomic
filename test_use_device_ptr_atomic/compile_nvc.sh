nvcc -c cuda_scatter_add.cu -arch=sm_70
nvc++ -c test_openmp.C -mp=gpu -cuda -Minfo=mp
nvc++ -c test_openmp_correctness.C -mp=gpu -cuda -Minfo=mp
nvc++ cuda_scatter_add.o test_openmp.o -mp=gpu -Minfo=mp -cuda
nvc++ cuda_scatter_add.o test_openmp_correctness.o -mp=gpu -Minfo=mp -cuda -o debug.out
