nvcc -O3 -c cuda_scatter_add.cu -arch=sm_70
clang++ -O3 -c -fopenmp -fopenmp-targets=nvptx64-nvidia -Xopenmp-target -march=sm_70 test_openmp.C
clang++ -O3 -c -fopenmp -fopenmp-targets=nvptx64-nvidia -Xopenmp-target -march=sm_70 test_openmp_correctness.C
clang++ -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia -Xopenmp-target -march=sm_70 -L/usr/local/cuda/lib64 -lcudart -ldl -lrt -pthread cuda_scatter_add.o test_openmp.o
clang++ -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia -Xopenmp-target -march=sm_70 -L/usr/local/cuda/lib64 -lcudart -ldl -lrt -pthread cuda_scatter_add.o test_openmp_correctness.o -o debug.out
