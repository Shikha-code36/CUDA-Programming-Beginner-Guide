# Chapter 1: Basics of CUDA Programming

## 1. CUDA Architecture

### GPUs (Graphics Processing Units)

- **Role**: GPUs are specialized processors designed for parallel computation. Unlike CPUs, which excel at sequential tasks, GPUs handle massive parallelism.
- **CUDA Cores**: GPUs consist of thousands of small processing units called CUDA cores. Each core can execute a thread independently.
- **Streaming Multiprocessors (SMs)**: SMs group CUDA cores together. They manage thread execution, memory access, and synchronization.
- **Warp**: A warp is the smallest unit of execution on a GPU. It consists of 32 threads that execute in lockstep.

## 2. Kernel Execution

### What is a Kernel?

- A **kernel** is a function written in CUDA C/C++ that runs on the GPU.
- Kernels are launched from the CPU and executed by multiple threads in parallel.

### Threads, Blocks, and Grids

- **Thread**: The smallest unit of execution within a kernel. Threads execute the same code but operate on different data.
- **Block**: Threads are organized into blocks. A block contains multiple threads that can synchronize and share data.
- **Grid**: A grid consists of multiple blocks. Blocks within a grid can run concurrently.

### Thread Indexing

- Each thread has a unique index within its block.
- You can access thread indices using `threadIdx.x`, `threadIdx.y`, and `threadIdx.z`.

## 3. Memory Hierarchy

### Global Memory

- **Role**: Global memory is the largest memory space accessible by both the CPU and GPU.
- **Usage**: Store data that needs to persist across kernel launches.
- **Access Time**: Relatively slow compared to other memory types.

### Shared Memory

- **Role**: Shared memory is a small, fast memory space shared among threads within a block.
- **Usage**: Store data that threads need to share or reuse during a block's execution.
- **Access Time**: Much faster than global memory.

### Constant Memory

- **Role**: Constant memory holds read-only data that remains constant during kernel execution.
- **Usage**: Store constants, lookup tables, or other data shared across threads.
- **Access Time**: Similar to shared memory.

### Local Memory

- **Role**: Local memory is private to each thread.
- **Usage**: Automatically allocated for local variables within a thread.
- **Access Time**: Slowest memory type; avoid excessive use.

## Conclusion

Understanding CUDA architecture, kernel execution, and memory hierarchy is crucial for efficient GPU programming. As you proceed with your projects, keep these concepts in mind, and explore optimization techniques to maximize performance.

