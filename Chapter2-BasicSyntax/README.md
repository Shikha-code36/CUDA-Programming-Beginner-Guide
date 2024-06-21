# Chapter 1: Basics of CUDA Programming Syntax

## 1. Including Headers

In C++ (and CUDA), we use headers to include predefined functions and classes. The most common header is `<iostream>`, which provides input/output functionality. For example:

```cpp
#include <iostream>
```

## 2. Data Types

### a. Integers (`int`, `short`, `long`)

- **`int`**: Represents whole numbers (e.g., 5, -10, 0).
- **`short`**: Short integers with smaller range.
- **`long`**: Long integers with larger range.

### b. Floating-Point Numbers (`float`, `double`)

- **`float`**: Single-precision floating-point (e.g., 3.14, -0.001).
- **`double`**: Double-precision floating-point (more accurate, but uses more memory).

### c. Other Types

- **`char`**: Represents individual characters (e.g., 'A', 'b', '$').
- **`bool`**: Represents true or false values.

## 3. Constants

- Use `const` to define constants (values that don't change during program execution).
- Example:

  ```cpp
  const float PI = 3.14159;
  ```

## 4. Loops

### a. `for` Loop

- Repeats a block of code a specified number of times.
- Example:

  ```cpp
  for (int i = 0; i < 10; ++i) {
      // Code to execute
  }
  ```

### b. `while` Loop

- Repeats a block of code while a condition is true.
- Example:

  ```cpp
  int count = 0;
  while (count < 5) {
      // Code to execute
      ++count;
  }
  ```

## 5. The `int main()` Function

- Every C++ program starts with the `main` function.
- It's the entry point of your program, where execution begins.
- The `int` before `main` indicates that the function returns an integer (usually 0 for successful execution).

## 6. Writing a Simple Kernel

- A **kernel** is a function that runs on the GPU.
- It's the heart of CUDA programming.
- Kernels are defined using the `__global__` keyword.

Example of a simple kernel:

```cpp
__global__ void vectorAdd(float* A, float* B, float* C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        C[tid] = A[tid] + B[tid];
    }
}
```

## 7. Thread Indexing

- In CUDA, we work with threads.
- Each thread has a unique index.
- We calculate the thread index using `blockIdx.x`, `blockDim.x`, and `threadIdx.x`.

## 8. Memory Allocation

- We allocate memory for data on both the CPU (host) and GPU (device).
- `cudaMalloc` allocates memory on the device.
- `cudaMemcpy` transfers data between host and device.

## 9. Launching the Kernel

- We launch the kernel using `<<<numBlocks, threadsPerBlock>>>` syntax.
- `numBlocks` and `threadsPerBlock` determine the grid and block dimensions.

## 10. Clean Up

- After using GPU memory, we free it using `cudaFree`.
- Also, delete any host memory allocated with `new`.