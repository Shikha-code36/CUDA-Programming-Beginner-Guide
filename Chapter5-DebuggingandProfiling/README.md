# Chapter 5: Debugging and Profiling CUDA Programs

In this chapter, we will explore various tools and techniques to debug and profile CUDA programs. Debugging helps identify and fix errors in your code, while profiling helps optimize performance by analyzing how your code executes on the GPU.

## Table of Contents

1. [Introduction](#introduction)
2. [Debugging CUDA Programs](#debugging-cuda-programs)
    - [Using cuda-gdb](#using-cuda-gdb)
    - [Using Nsight Eclipse Edition](#using-nsight-eclipse-edition)
    - [Using Nsight Visual Studio Edition](#using-nsight-visual-studio-edition)
3. [Profiling CUDA Programs](#profiling-cuda-programs)
    - [Using Nsight Systems](#using-nsight-systems)
    - [Using Nsight Compute](#using-nsight-compute)
4. [Sample Code](#sample-code)
5. [Additional Resources](#additional-resources)

## Introduction

Debugging and profiling are essential steps in CUDA development. They help ensure your code runs correctly and efficiently on the GPU. This guide will introduce you to the tools and techniques needed to debug and profile CUDA programs effectively.

## Debugging CUDA Programs

### Using cuda-gdb

`cuda-gdb` is a powerful debugger for CUDA applications. It allows you to set breakpoints, step through code, and inspect variables.

1. **Compile with Debug Information**:
    ```sh
    nvcc -g -G -o debapro debapro.cu
    ```

2. **Start cuda-gdb**:
    ```sh
    cuda-gdb ./debapro
    ```

3. **Set Breakpoints**:
    ```sh
    (cuda-gdb) break main
    ```

4. **Run the Program**:
    ```sh
    (cuda-gdb) run
    ```

5. **Step Through Code**:
    ```sh
    (cuda-gdb) next
    ```

6. **Inspect Variables**:
    ```sh
    (cuda-gdb) print variable_name
    ```

### Using Nsight Eclipse Edition

Nsight Eclipse Edition provides an integrated development environment for debugging CUDA applications.

1. **Open Nsight Eclipse Edition**.
2. **Create a Debug Configuration**:
    - Go to **Run > Debug Configurations**.
    - Create a new CUDA C/C++ Application configuration.
    - Set the project and application to your `debapro` executable.
3. **Set Breakpoints and Start Debugging**.

### Using Nsight Visual Studio Edition

Nsight Visual Studio Edition integrates with Visual Studio to provide debugging capabilities for CUDA applications.

1. **Open Visual Studio**.
2. **Create a Debug Configuration**:
    - Go to **Debug > Attach to Process**.
    - Select the CUDA application you want to debug.
3. **Set Breakpoints and Start Debugging**.

## Profiling CUDA Programs

### Using Nsight Systems

Nsight Systems provides system-wide performance analysis, helping you identify bottlenecks in your application.

1. **Run Nsight Systems**:
    ```sh
    nsys profile ./debapro
    ```

2. **Analyze the Report**:
    - This command generates a report file (e.g., `report.qdrep`).
    - Open this file in the Nsight Systems GUI for detailed analysis.

### Using Nsight Compute

Nsight Compute provides detailed analysis of CUDA kernel performance.

1. **Run Nsight Compute**:
    ```sh
    ncu ./debapro
    ```

2. **Analyze the Output**:
    - Nsight Compute will provide detailed metrics about the kernel execution.

## Sample Code

Here is a sample CUDA program for matrix multiplication:

```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 16  // Size of the matrix (N x N)

__global__ void matrixMul(const float *A, const float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0;

    if (row < width && col < width) {
        for (int k = 0; k < width; ++k) {
            value += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = value;
    }
}

int main() {
    int size = N * N * sizeof(float);
    float h_A[N * N], h_B[N * N], h_C[N * N];

    // Initialize matrices
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += h_A[i * N + k] * h_B[k * N + j];
            }
            if (fabs(sum - h_C[i * N + j]) > 1e-5) {
                std::cerr << "Result verification failed at element (" << i << ", " << j << ")" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    std::cout << "Test PASSED" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```

## Resources to check

- [NVIDIA Developer Blog on Profiling and Debugging CUDA Applications](https://developer.nvidia.com/blog/new-video-tutuorial-profiling-and-debugging-nvidia-cuda-applications/)²
- [CUDA Profiler User’s Guide](https://docs.nvidia.com/cuda/profiler-users-guide/index.html)⁴
- [NVIDIA Nsight Developer Tools](https://developer.nvidia.com/nsight-visual-studio-edition)⁵
