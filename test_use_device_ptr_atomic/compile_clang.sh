#clang++ -c cuda_scatter_add.cu --cuda-gpu-arch=sm_70
nvcc -c cuda_scatter_add.cu -arch=sm_70
clang++ -c -fopenmp -fopenmp-targets=nvptx64-nvidia -Xopenmp-target -march=sm_70 test_openmp.C
clang++ -c -fopenmp -fopenmp-targets=nvptx64-nvidia -Xopenmp-target -march=sm_70 test_openmp_correctness.C
clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia -Xopenmp-target -march=sm_70 -L/usr/local/cuda/lib64 -lcudart -ldl -lrt -pthread cuda_scatter_add.o test_openmp.o
clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia -Xopenmp-target -march=sm_70 -L/usr/local/cuda/lib64 -lcudart -ldl -lrt -pthread cuda_scatter_add.o test_openmp_correctness.o -o debug.out
