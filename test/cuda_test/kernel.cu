#include <cuda_runtime.h>

__global__ void matmul_gpu(float* A, float* B, float* C, int M, int K, int N) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < N && row < M) {
        float sum = 0.0f;

        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }

        C[row * N + col] += sum;
    }
}