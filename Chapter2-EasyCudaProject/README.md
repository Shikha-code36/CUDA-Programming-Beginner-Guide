# Chapter 2: Simple CUDA Vector Addition

## Overview

In this chapter, we'll create a basic CUDA program that performs vector addition using the GPU. The program consists of a CUDA kernel that adds corresponding elements from two input vectors and stores the result in an output vector.

### Code Explanation

Let's go through the code step by step:

1. **Kernel Function (`vectorAdd`)**:
   - The `vectorAdd` function is the heart of our CUDA program. It runs on the GPU and performs the vector addition.
   - It takes four arguments:
     - `A`: Pointer to the first input vector.
     - `B`: Pointer to the second input vector.
     - `C`: Pointer to the output vector (where the result will be stored).
     - `size`: Size of the vectors (number of elements).
   - Inside the kernel, each thread computes the sum of corresponding elements from `A` and `B` and stores the result in `C`.

2. **Main Function**:
   - The `main` function sets up the host (CPU) and device (GPU) memory, initializes input vectors, launches the kernel, and retrieves the result.
   - Key steps in the `main` function:
     - Allocate memory for vectors (`h_A`, `h_B`, and `h_C`) on the host.
     - Initialize input vectors (`h_A` and `h_B`) with sample values.
     - Allocate memory for vectors (`d_A`, `d_B`, and `d_C`) on the device (GPU).
     - Copy data from host to device using `cudaMemcpy`.
     - Launch the `vectorAdd` kernel with appropriate block and grid dimensions.
     - Copy the result back from the device to the host.
     - Print the result (output vector `h_C`).

3. **Memory Allocation and Transfer**:
   - We allocate memory for vectors on both the host and the device.
   - `cudaMalloc` allocates memory on the device.
   - `cudaMemcpy` transfers data between host and device.

4. **Kernel Launch**:
   - We launch the `vectorAdd` kernel using `<<<numBlocks, threadsPerBlock>>>` syntax.
   - `numBlocks` and `threadsPerBlock` determine the grid and block dimensions.

5. **Clean Up**:
   - We free the allocated device memory using `cudaFree`.
   - We also delete the host vectors (`h_A`, `h_B`, and `h_C`) to avoid memory leaks.

## How to Compile and Run

1. **Compile the Code**:
   - Open your terminal or command prompt.
   - Navigate to the folder containing `vector_addition.cu`.
   - Compile the code using `nvcc` (NVIDIA CUDA Compiler):
     ```
     nvcc vector_addition.cu -o vector_addition
     ```
    ![Compile Code Image](https://github.com/Shikha-code36/CUDA-Programming-Beginner-Guide/blob/main/Chapter2-EasyCudaProject/part1.png)

2. **Run the Executable**:
   - Execute the compiled binary:
     ```
     ./vector_addition
     ```
   - You'll see the result of vector addition printed to the console.

   ![Executable Code Image](https://github.com/Shikha-code36/CUDA-Programming-Beginner-Guide/blob/main/Chapter2-EasyCudaProject/part2.png)

