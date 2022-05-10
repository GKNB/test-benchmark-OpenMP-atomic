nvcc test_cuda.cu -Xcompiler -fopenmp -lgomp -arch=sm_70 -o run_cuda_nvcc
#clang++ --cuda-gpu-arch=sm_70 -L/usr/local/cuda/lib64 -lcudart_static -ldl -lrt -pthread -fopenmp test_cuda.cu -o run_cuda 
clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia -Xopenmp-target -march=sm_70 test_openmp.C -o run_openmp
