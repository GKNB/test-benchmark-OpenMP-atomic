hipcc -O3 test_hip.C -fopenmp -o run_hip
/opt/rocm/llvm/bin/clang++ -O3 -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906 test_openmp.C -o run_openmp

