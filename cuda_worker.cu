#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>

#include "utils.h"

#include "cuda_worker.cuh"

constexpr int BLOCK_SIZE = 8;
// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// template <int BLOCK_SIZE>
__global__ void apply_von_neuman_boundary_slice(int N, double *source, double *curr, double *next, float delta) {
    // Calculate 3D indices for the current thread
    int block_k = blockIdx.z * BLOCK_SIZE + threadIdx.z;
    int block_j = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int block_i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (block_k >= N || block_j >= N || block_i >= N) return; // Ensure within bounds

    // Apply boundary conditions (as per your logic)
    // Example for boundaries as in your original function
    if (block_j == 0 && block_i == 0) {
        idx(next, N, block_k, 0, 0) = (2 * idx(curr, N, block_k, 0, 0 + 1)
            + 2 * idx(curr, N, block_k, 0 + 1, 0)
            + idx(curr, N, block_k + 1, 0, 0) + idx(curr, N, block_k - 1, 0, 0)
            - delta * delta * idx(source, N, block_k, 0, 0)) / 6;
    }

    if (block_j == N && block_j == N) {
        idx(next, N, block_k, N - 1, N - 1) = (2 * idx(curr, N, block_k, N - 1, N - 1 - 1)
            + 2 * idx(curr, N, block_k, N - 1 - 1, N - 1)
            + idx(curr, N, block_k + 1, N - 1, N - 1) + idx(curr, N, block_k - 1, N - 1, N - 1)
            - delta * delta * idx(source, N, block_k, N - 1, N - 1)) / 6;
    }

    if (block_j == N && block_i == 0) {
        idx(next, N, block_k, N - 1, 0) = (2 * idx(curr, N, block_k, N - 1, 0 + 1)
            + 2 * idx(curr, N, block_k, N - 1 - 1, 0)
            + idx(curr, N, block_k + 1, N - 1, 0) + idx(curr, N, block_k - 1, N - 1, 0)
            - delta * delta * idx(source, N, block_k, N - 1, 0)) / 6;
    }

    if (block_j == 0 && block_j == N) {
        idx(next, N, block_k, 0, N - 1) = (2 * idx(curr, N, block_k, 0, N - 1 - 1)
            + 2 * idx(curr, N, block_k, 0 + 1, N - 1)
            + idx(curr, N, block_k + 1, 0, N - 1) + idx(curr, N, block_k - 1, 0, N - 1)
            - delta * delta * idx(source, N, block_k, 0, N - 1)) / 6;
    }

    if (block_i == 0) {
        for (int j = block_j + 1; j < block_j - 1; j++) {
            idx(next, N, block_k, j, 0) = (2 * idx(curr, N, block_k, j, 0 + 1)
                + idx(curr, N, block_k, j + 1, 0) + idx(curr, N, block_k, j - 1, 0)
                + idx(curr, N, block_k + 1, j, 0) + idx(curr, N, block_k - 1, j, 0)
                - delta * delta * idx(source, N, block_k, j, 0)) / 6;
        }
    }

    if (block_i == N) {
        for (int j = block_j + 1; j < block_j - 1; j++) {
            idx(next, N, block_k, j, N - 1) = (2 * idx(curr, N, block_k, j, N - 1 - 1)
                + idx(curr, N, block_k, j + 1, N - 1) + idx(curr, N, block_k, j - 1, N - 1)
                + idx(curr, N, block_k + 1, j, N - 1) + idx(curr, N, block_k - 1, j, N - 1)
                - delta * delta * idx(source, N, block_k, j, N - 1)) / 6;
        }
    }

    if (block_j == 0) {
        for (int i = block_i + 1; i < block_i - 1; i++) {
            idx(next, N, block_k, 0, i) = (idx(curr, N, block_k, 0, i + 1) + idx(curr, N, block_k, 0, i - 1)
                + 2 * idx(curr, N, block_k, 0 + 1, i)
                + idx(curr, N, block_k + 1, 0, i) + idx(curr, N, block_k - 1, 0, i)
                - delta * delta * idx(source, N, block_k, 0, i)) / 6;
        }
    }

    if (block_j == N) {
        for (int i = block_i + 1; i < block_i - 1; i++) {
            idx(next, N, block_k, N - 1, i) = (idx(curr, N, block_k, N - 1, i + 1) + idx(curr, N, block_k, N - 1, i - 1)
                + 2 * idx(curr, N, block_k, N - 1 - 1, i)
                + idx(curr, N, block_k + 1, N - 1, i) + idx(curr, N, block_k - 1, N - 1, i)
                - delta * delta * idx(source, N, block_k, N - 1, i)) / 6;
        }
    }

    // Other boundary conditions go here
}

// template <int BLOCK_SIZE>
__global__ void poisson_iteration_inner_slice(int N, double *source, double *curr, double *next, float delta) {
    // Calculate 3D indices for the current thread
    int k = blockIdx.z * BLOCK_SIZE + threadIdx.z;
    int j = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (k >= N || j >= N || i >= N) return; // Ensure within bounds

    // Perform the inner iteration logic
    idx(next, N, k, j, i) = (idx(curr, N, k, j, i + 1) + idx(curr, N, k, j, i - 1)
                            + idx(curr, N, k, j + 1, i) + idx(curr, N, k, j - 1, i)
                            + idx(curr, N, k + 1, j, i) + idx(curr, N, k - 1, j, i)
                            - delta * delta * idx(source, N, k, j, i)) / 6;
}

// void* worker_thread(void* pargs) {
//     workerThread_t* worker_info = (workerThread_t*)pargs;
//     int N = worker_info->N;

//     // Device memory allocation
//     double *d_source, *d_curr, *d_next;
//     cudaError_t err;

//     err = cudaMalloc((void**)&d_source, N * N * N * sizeof(double));
//     if (err != cudaSuccess) {
//         fprintf(stderr, "CUDA error (malloc d_source): %s\n", cudaGetErrorString(err));
//         return NULL;
//     }
//     err = cudaMalloc((void**)&d_curr, N * N * N * sizeof(double));
//     if (err != cudaSuccess) {
//         fprintf(stderr, "CUDA error (malloc d_curr): %s\n", cudaGetErrorString(err));
//         cudaFree(d_source);
//         return NULL;
//     }
//     err = cudaMalloc((void**)&d_next, N * N * N * sizeof(double));
//     if (err != cudaSuccess) {
//         fprintf(stderr, "CUDA error (malloc d_next): %s\n", cudaGetErrorString(err));
//         cudaFree(d_source);
//         cudaFree(d_curr);
//         return NULL;
//     }

//     // Copy data to device
//     err = cudaMemcpy(d_source, worker_info->source, N * N * N * sizeof(double), cudaMemcpyHostToDevice);
//     if (err != cudaSuccess) {
//         fprintf(stderr, "CUDA error (memcpy to d_source): %s\n", cudaGetErrorString(err));
//         cudaFree(d_source);
//         cudaFree(d_curr);
//         cudaFree(d_next);
//         return NULL;
//     }

//     // Define kernel launch configuration
//     dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
//     dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

//     // Create a CUDA stream
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);

//     for (int n = 0; n < worker_info->iterations; n++) {
//         // Call the boundary condition kernel
//         apply_von_neuman_boundary_slice<BLOCK_SIZE><<<numBlocks, threadsPerBlock, 0, stream>>>(N, d_source, d_curr, d_next, worker_info->delta);

//         // Call the inner iteration kernel
//         poisson_iteration_inner_slice<BLOCK_SIZE><<<numBlocks, threadsPerBlock, 0, stream>>>(N, d_source, d_curr, d_next, worker_info->delta);

//         // Check for errors after kernel launches
//         err = cudaGetLastError();
//         if (err != cudaSuccess) {
//             fprintf(stderr, "CUDA error (kernel launch): %s\n", cudaGetErrorString(err));
//             break;
//         }

//         // Synchronize before copying data back
//         cudaStreamSynchronize(stream);

//         // Memory copy back to host if needed
//         err = cudaMemcpy(worker_info->curr, d_next, N * N * N * sizeof(double), cudaMemcpyDeviceToHost);
//         if (err != cudaSuccess) {
//             fprintf(stderr, "CUDA error (memcpy to host): %s\n", cudaGetErrorString(err));
//             break;
//         }
//     }

//     // Clean up
//     cudaStreamDestroy(stream);
//     cudaFree(d_source);
//     cudaFree(d_curr);
//     cudaFree(d_next);

//     return NULL;
// }
