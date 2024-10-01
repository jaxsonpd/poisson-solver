#include <stdint.h>
#include <stdbool.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

// // template <int BLOCK_SIZE>
// __global__ void apply_von_neuman_boundary_slice(int N, double *source, double *curr, double *next, float delta);

// // template <int BLOCK_SIZE>
// __global__ void poisson_iteration_inner_slice(int N, double *source, double *curr, double *next, float delta);

__global__ void poisson_slice(int N, double *source, double *curr, double *next, float delta);