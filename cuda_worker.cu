#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>

#include "utils.h"

#include "cuda_worker.cuh"

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// // template <int BLOCK_SIZE>
// __global__ void apply_von_neuman_boundary_slice(int N, double *source, double *curr, double *next, float delta) {
//     // Calculate 3D indices for the current thread
//     int block_k = blockIdx.z * blockDim.z + threadIdx.z;
//     int block_j = blockIdx.y * blockDim.y + threadIdx.y;
//     int block_i = blockIdx.x * blockDim.x + threadIdx.x;

//     if (block_k >= N || block_j >= N || block_i >= N) return; // Ensure within bounds

//     if (block_k == 0 || block_k == N-1) return; // Maintain constant bounds

//     // Apply boundary conditions
//     if (block_j == 0 && block_i == 0) {
//         idx(next, N, block_k, 0, 0) = (2 * idx(curr, N, block_k, 0, 0 + 1)
//             + 2 * idx(curr, N, block_k, 0 + 1, 0)
//             + idx(curr, N, block_k + 1, 0, 0) + idx(curr, N, block_k - 1, 0, 0)
//             - delta * delta * idx(source, N, block_k, 0, 0)) / 6;
//     }

//     if (block_j == N-1 && block_i == N-1) {
//         idx(next, N, block_k, N - 1, N - 1) = (2 * idx(curr, N, block_k, N - 1, N - 1 - 1)
//             + 2 * idx(curr, N, block_k, N - 1 - 1, N - 1)
//             + idx(curr, N, block_k + 1, N - 1, N - 1) + idx(curr, N, block_k - 1, N - 1, N - 1)
//             - delta * delta * idx(source, N, block_k, N - 1, N - 1)) / 6;
//     }

//     if (block_j == N-1 && block_i == 0) {
//         idx(next, N, block_k, N - 1, 0) = (2 * idx(curr, N, block_k, N - 1, 0 + 1)
//             + 2 * idx(curr, N, block_k, N - 1 - 1, 0)
//             + idx(curr, N, block_k + 1, N - 1, 0) + idx(curr, N, block_k - 1, N - 1, 0)
//             - delta * delta * idx(source, N, block_k, N - 1, 0)) / 6;
//     }

//     if (block_j == 0 && block_i == N-1) {
//         idx(next, N, block_k, 0, N - 1) = (2 * idx(curr, N, block_k, 0, N - 1 - 1)
//             + 2 * idx(curr, N, block_k, 0 + 1, N - 1)
//             + idx(curr, N, block_k + 1, 0, N - 1) + idx(curr, N, block_k - 1, 0, N - 1)
//             - delta * delta * idx(source, N, block_k, 0, N - 1)) / 6;
//     }

//     if (block_i == 0) {
//         idx(next, N, block_k, block_j, 0) = (2 * idx(curr, N, block_k, block_j, 0 + 1)
//             + idx(curr, N, block_k, block_j + 1, 0) + idx(curr, N, block_k, block_j - 1, 0)
//             + idx(curr, N, block_k + 1, block_j, 0) + idx(curr, N, block_k - 1, block_j, 0)
//             - delta * delta * idx(source, N, block_k, block_j, 0)) / 6;
//     }

//     if (block_i == N-1) {
//         idx(next, N, block_k, block_j, N - 1) = (2 * idx(curr, N, block_k, block_j, N - 1 - 1)
//             + idx(curr, N, block_k, block_j + 1, N - 1) + idx(curr, N, block_k, block_j - 1, N - 1)
//             + idx(curr, N, block_k + 1, block_j, N - 1) + idx(curr, N, block_k - 1, block_j, N - 1)
//             - delta * delta * idx(source, N, block_k, block_j, N - 1)) / 6;
//     }

//     if (block_j == 0) {
//         idx(next, N, block_k, 0, block_i) = (idx(curr, N, block_k, 0, block_i + 1) + idx(curr, N, block_k, 0, block_i - 1)
//             + 2 * idx(curr, N, block_k, 0 + 1, block_i)
//             + idx(curr, N, block_k + 1, 0, block_i) + idx(curr, N, block_k - 1, 0, block_i)
//             - delta * delta * idx(source, N, block_k, 0, block_i)) / 6;
//     }

//     if (block_j == N-1) {
//         idx(next, N, block_k, N - 1, block_i) = (idx(curr, N, block_k, N - 1, block_i + 1) + idx(curr, N, block_k, N - 1, block_i - 1)
//             + 2 * idx(curr, N, block_k, N - 1 - 1, block_i)
//             + idx(curr, N, block_k + 1, N - 1, block_i) + idx(curr, N, block_k - 1, N - 1, block_i)
//             - delta * delta * idx(source, N, block_k, N - 1, block_i)) / 6;
//     }
//     // printf("%d %d %d\n",block_i,block_j,block_k);
//     __syncthreads();
// }

// // template <int BLOCK_SIZE>s
// __global__ void poisson_iteration_inner_slice(int N, double *source, double *curr, double *next, float delta) {
//     // Calculate 3D indices for the current thread
//     int k = blockIdx.z * blockDim.z + threadIdx.z;
//     int j = blockIdx.y * blockDim.y + threadIdx.y;
//     int i = blockIdx.x * blockDim.x + threadIdx.x;

//     if (k >= N || j >= N || i >= N) return; // Ensure within bounds

//     if (k == 0 || k == N-1) return; // Maintain constant bounds

//     // Perform the inner iteration logic
//     idx(next, N, k, j, i) = (idx(curr, N, k, j, i + 1) + idx(curr, N, k, j, i - 1)
//                             + idx(curr, N, k, j + 1, i) + idx(curr, N, k, j - 1, i)
//                             + idx(curr, N, k + 1, j, i) + idx(curr, N, k - 1, j, i)
//                             - delta * delta * idx(source, N, k, j, i)) / 6;
// }

__global__ void poisson_slice(int N, double *source, double *curr, double *next, float delta) {
    // Calculate 3D indices for the current thread
    int block_k = blockIdx.z * blockDim.z + threadIdx.z;
    int block_j = blockIdx.y * blockDim.y + threadIdx.y;
    int block_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_k >= N || block_j >= N || block_i >= N) return; // Ensure within bounds

    if (block_k == 0 || block_k == N-1) return; // Maintain constant bounds

    // Apply boundary conditions
    if (block_j == 0 && block_i == 0) {
        idx(next, N, block_k, 0, 0) = (2 * idx(curr, N, block_k, 0, 0 + 1)
            + 2 * idx(curr, N, block_k, 0 + 1, 0)
            + idx(curr, N, block_k + 1, 0, 0) + idx(curr, N, block_k - 1, 0, 0)
            - delta * delta * idx(source, N, block_k, 0, 0)) / 6;
        return;
    }

    if (block_j == N-1 && block_i == N-1) {
        idx(next, N, block_k, N - 1, N - 1) = (2 * idx(curr, N, block_k, N - 1, N - 1 - 1)
            + 2 * idx(curr, N, block_k, N - 1 - 1, N - 1)
            + idx(curr, N, block_k + 1, N - 1, N - 1) + idx(curr, N, block_k - 1, N - 1, N - 1)
            - delta * delta * idx(source, N, block_k, N - 1, N - 1)) / 6;
        return;
    }

    if (block_j == N-1 && block_i == 0) {
        idx(next, N, block_k, N - 1, 0) = (2 * idx(curr, N, block_k, N - 1, 0 + 1)
            + 2 * idx(curr, N, block_k, N - 1 - 1, 0)
            + idx(curr, N, block_k + 1, N - 1, 0) + idx(curr, N, block_k - 1, N - 1, 0)
            - delta * delta * idx(source, N, block_k, N - 1, 0)) / 6;
        return;
    }

    if (block_j == 0 && block_i == N-1) {
        idx(next, N, block_k, 0, N - 1) = (2 * idx(curr, N, block_k, 0, N - 1 - 1)
            + 2 * idx(curr, N, block_k, 0 + 1, N - 1)
            + idx(curr, N, block_k + 1, 0, N - 1) + idx(curr, N, block_k - 1, 0, N - 1)
            - delta * delta * idx(source, N, block_k, 0, N - 1)) / 6;
        return;
    }

    if (block_i == 0) {
        idx(next, N, block_k, block_j, 0) = (2 * idx(curr, N, block_k, block_j, 0 + 1)
            + idx(curr, N, block_k, block_j + 1, 0) + idx(curr, N, block_k, block_j - 1, 0)
            + idx(curr, N, block_k + 1, block_j, 0) + idx(curr, N, block_k - 1, block_j, 0)
            - delta * delta * idx(source, N, block_k, block_j, 0)) / 6;
        return;
    }

    if (block_i == N-1) {
        idx(next, N, block_k, block_j, N - 1) = (2 * idx(curr, N, block_k, block_j, N - 1 - 1)
            + idx(curr, N, block_k, block_j + 1, N - 1) + idx(curr, N, block_k, block_j - 1, N - 1)
            + idx(curr, N, block_k + 1, block_j, N - 1) + idx(curr, N, block_k - 1, block_j, N - 1)
            - delta * delta * idx(source, N, block_k, block_j, N - 1)) / 6;
        return;
    }

    if (block_j == 0) {
        idx(next, N, block_k, 0, block_i) = (idx(curr, N, block_k, 0, block_i + 1) + idx(curr, N, block_k, 0, block_i - 1)
            + 2 * idx(curr, N, block_k, 0 + 1, block_i)
            + idx(curr, N, block_k + 1, 0, block_i) + idx(curr, N, block_k - 1, 0, block_i)
            - delta * delta * idx(source, N, block_k, 0, block_i)) / 6;
        return;
    }

    if (block_j == N-1) {
        idx(next, N, block_k, N - 1, block_i) = (idx(curr, N, block_k, N - 1, block_i + 1) + idx(curr, N, block_k, N - 1, block_i - 1)
            + 2 * idx(curr, N, block_k, N - 1 - 1, block_i)
            + idx(curr, N, block_k + 1, N - 1, block_i) + idx(curr, N, block_k - 1, N - 1, block_i)
            - delta * delta * idx(source, N, block_k, N - 1, block_i)) / 6;
        return;
    } 

    idx(next, N, block_k, block_j, block_i) = (idx(curr, N, block_k, block_j, block_i + 1) + idx(curr, N, block_k, block_j, block_i - 1)
                            + idx(curr, N, block_k, block_j + 1, block_i) + idx(curr, N, block_k, block_j - 1, block_i)
                            + idx(curr, N, block_k + 1, block_j, block_i) + idx(curr, N, block_k - 1, block_j, block_i)
                            - delta * delta * idx(source, N, block_k, block_j, block_i)) / 6;
}