# Chapter 4: Intermediate Algorithms - CUDA Merge Sort

## Overview

This chapter explores parallelizable algorithms, focusing on sorting techniques. We have chosen to implement merge sort using CUDA as an example of an intermediate-level algorithm that can benefit from parallel processing.

## Merge Sort in CUDA

Merge sort is an efficient, stable sorting algorithm that follows the divide-and-conquer paradigm. It's well-suited for parallel implementation due to its recursive nature and the independence of its sub-problems.

### Key Components

1. `merge` function (device):
   - Merges two sorted subarrays into a single sorted array.
   - Uses temporary storage for merging to avoid in-place operations.

2. `mergeSortRecursive` function (device):
   - Implements the recursive part of the merge sort algorithm.
   - Divides the array, recursively sorts subarrays, and merges results.

3. `mergeSort` kernel (global):
   - Entry point for the GPU execution.
   - Calls the recursive merge sort function.

4. `main` function:
   - Sets up the problem on the host.
   - Allocates memory on the device.
   - Launches the kernel and retrieves results.

## Implementation Details

- The code uses CUDA-specific keywords like `__device__` and `__global__` to define functions that run on the GPU.
- Memory is allocated on both host and device to facilitate data transfer.
- The sorting is performed entirely on the GPU, with only the final sorted array transferred back to the host.

## Optimization Techniques

While this implementation demonstrates the basic structure of a CUDA merge sort, several optimization techniques could be applied:

1. Shared Memory: Utilize shared memory for faster access to frequently used data within a thread block.
2. Coalesced Memory Access: Optimize global memory access patterns for better performance.
3. Dynamic Parallelism: Use CUDA's dynamic parallelism to launch child kernels from within the device code, potentially improving the parallelization of the recursive steps.
4. Hybrid Approach: Combine GPU parallelism for large-scale divisions with CPU processing for smaller subarrays.

## Execution
![Merge](https://github.com/Shikha-code36/CUDA-Programming-Beginner-Guide/blob/main/Chapter4-IntermediateAlgo/merge.png)

## Conclusion

This CUDA implementation of merge sort demonstrates how a classic sorting algorithm can be adapted for parallel execution on a GPU. It serves as a foundation for understanding more complex parallel algorithms and optimization techniques in CUDA programming.