#include <iostream>

// Kernel for vector addition
__global__ void vectorAdd(float* A, float* B, float* C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        C[tid] = A[tid] + B[tid];
    }
}

int main() {
    const int N = 1024; // Size of vectors
    const int threadsPerBlock = 256;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate memory for vectors on the host
    float* h_A = new float[N];
    float* h_B = new float[N];
    float* h_C = new float[N];

    // Initialize vectors with some values (you can modify this)
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // Allocate memory for vectors on the device (GPU)
    float* d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel
    vectorAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Print the result (you can modify this)
    for (int i = 0; i < N; ++i) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
