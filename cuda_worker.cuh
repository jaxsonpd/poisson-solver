#include <stdint.h>
#include <stdbool.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

/**
 * @brief perform one item update of the poisson cube as part of a Cuda kernel
 * 
 * @param N the size of the cube
 * @param source the source array
 * @param curr the current array
 * @param next the next array
 * @param delta the distance between cubes
 */
__global__ void poisson_slice(int N, double *source, double *curr, double *next, float delta);