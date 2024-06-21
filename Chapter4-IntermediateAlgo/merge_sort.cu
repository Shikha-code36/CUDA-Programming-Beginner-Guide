#include <iostream>
#include <cstdlib>

// Merge two sorted arrays
__device__ void merge(int* arr, int* temp, int left, int mid, int right) {
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right) {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i <= mid)
        temp[k++] = arr[i++];
    while (j <= right)
        temp[k++] = arr[j++];

    for (int idx = left; idx <= right; ++idx)
        arr[idx] = temp[idx];
}


// Recursive merge sort
__device__ void mergeSortRecursive(int* arr, int* temp, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSortRecursive(arr, temp, left, mid);
        mergeSortRecursive(arr, temp, mid + 1, right);
        merge(arr, temp, left, mid, right);
    }
}

// Kernel function to start the merge sort
__global__ void mergeSort(int* arr, int* temp, int N) {
    mergeSortRecursive(arr, temp, 0, N - 1);
}

int main() {
    const int N = 10; // Number of elements
    int arr[N] = {9, 3, 7, 1, 5, 8, 2, 4, 6, 0};
    int* d_arr;
    int* d_temp;

    cudaMalloc(&d_arr, N * sizeof(int));
    cudaMalloc(&d_temp, N * sizeof(int));
    cudaMemcpy(d_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice);

    mergeSort<<<1, 1>>>(d_arr, d_temp, N);

    cudaMemcpy(arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sorted array: ";
    for (int i = 0; i < N; ++i)
        std::cout << arr[i] << " ";
    std::cout << std::endl;

    cudaFree(d_arr);
    cudaFree(d_temp);

    return 0;
}