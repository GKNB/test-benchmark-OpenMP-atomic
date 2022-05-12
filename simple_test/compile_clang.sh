nvcc -O3 test_cuda.cu -Xcompiler -fopenmp -lgomp -arch=sm_70 -o run_cuda_nvcc
clang++ -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia -Xopenmp-target -march=sm_70 test_openmp.C -o run_openmp
