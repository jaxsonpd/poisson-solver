#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>

#include "utils.h"

#include "cuda_worker.cuh"

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define PRECISION double
#define BLOCK_SIZE 8

extern char* optarg;

// Global flag
// Set to true when operating in debug mode to enable verbose logging
static bool debug = false;

// Statics
const double top_boundary_cond = -1; // V The top dirlec boundary condition
const double bottom_boundary_cond = 1; // V The bottom dirlec boundary condition

void apply_const_boundary(int N, double* next) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            idx(next, N, 0, j, i) = top_boundary_cond;
            idx(next, N, N - 1, j, i) = bottom_boundary_cond;
        }
    }
}

/**
 * @brief Solve Poissons equation for a given cube with Dirichlet boundary
 * conditions on all sides.
 *
 * @param N             The edge length of the cube. n^3 number of elements.
 * @param source        Pointer to the source term cube, a.k.a. forcing function.
 * @param iterations    Number of iterations to perform.
 * @param threads       Number of threads to use for solving.
 * @param delta         Grid spacing.
 * @return double*      Solution to Poissons equation.  Caller must free.
 */
double* poisson_mixed(int N, double* source, int iterations, float delta) {
    if (debug) {
        printf("Starting solver with:\n"
            "n = %i\n"
            "iterations = %i\n"
            "delta = %f\n",
            N, iterations, delta);
    }

    // Allocate memory for the solution on the host
    double* curr = (double*)calloc(N * N * N, sizeof(double));
    double* next = (double*)calloc(N * N * N, sizeof(double));
    // Ensure we haven't run out of memory
    if (curr == NULL || next == NULL) {
        fprintf(stderr, "Error: ran out of memory when trying to allocate %i sized cube\n", N);
        exit(EXIT_FAILURE);
    }

    // Apply constant boundary
    apply_const_boundary(N, next);

    // Allocate device memory
    double *d_source, *d_curr, *d_next;
    cudaError_t ex;
    ex = cudaMalloc((void**)&d_source, N * N * N * sizeof(double));
    if (ex != 0) {
        while(1);
    }
    ex = cudaMalloc((void**)&d_curr, N * N * N * sizeof(double));
    if (ex != 0) {
        while(1);
    }
    ex = cudaMalloc((void**)&d_next, N * N * N * sizeof(double));
    if (ex != 0) {
        while(1);
    }

    // Copy data to device
    cudaMemcpy(d_source, source, N * N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_curr, next, N * N * N * sizeof(double), cudaMemcpyHostToDevice);

    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; // Calculate grid size

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(gridSize, gridSize, gridSize);

    // Main iteration loop
    for (int iter = 0; iter < iterations; iter++) {
        poisson_slice<<<numBlocks, threadsPerBlock>>>(N, d_source, d_curr, d_next, delta);

        // Swap pointers
        double* temp = d_curr;
        d_curr = d_next;
        d_next = temp;
    }

    // Copy the result back to the host
    cudaMemcpy(curr, d_curr, N * N * N * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_source);
    cudaFree(d_curr);
    cudaFree(d_next);
    free(next); // free the next buffer as it's no longer needed

    if (debug) {
        printf("Finished solving.\n");
    }

    return curr; // Return the result
}

int main(int argc, char** argv) {
    // Default settings for solver
    int iterations = 300;
    int n = 5;
    int threads = 3;
    float delta = 1;
    int x = -1;
    int y = -1;
    int z = -1;
    double amplitude = 1.0;

    int opt;

    // parse the command line arguments
    while ((opt = getopt(argc, argv, "h:n:i:x:y:z:a:t:d:")) != -1) {
        switch (opt) {
        case 'h':
            printf("Usage: poisson [-n size] [-x source x-poisition] [-y source y-position] [-z source z-position] [-a source amplitude] [-i iterations] [-t threads] [-d] (for debug mode)\n");
            return EXIT_SUCCESS;
        case 'n':
            n = atoi(optarg);
            break;
        case 'i':
            iterations = atoi(optarg);
            break;
        case 'x':
            x = atoi(optarg);
            break;
        case 'y':
            y = atoi(optarg);
            break;
        case 'z':
            z = atoi(optarg);
            break;
        case 'a':
            amplitude = atof(optarg);
            break;
        case 't':
            threads = atoi(optarg);
            break;
        case 'd':
            debug = true;
            break;
        default:
            fprintf(stderr, "Usage: poisson [-n size] [-x source x-poisition] [-y source y-position] [-z source z-position] [-a source amplitude]  [-i iterations] [-t threads] [-d] (for debug mode)\n");
            exit(EXIT_FAILURE);
        }
    }

    // Ensure we have an odd sized cube
    if (n % 2 == 0) {
        fprintf(stderr, "Error: n should be an odd number!\n");
        return EXIT_FAILURE;
    }

    // Create a source term with a single point in the centre
    double* source = (double*)calloc(n * n * n, sizeof(double));
    if (source == NULL) {
        fprintf(stderr, "Error: failed to allocated source term (n=%i)\n", n);
        return EXIT_FAILURE;
    }

    // Default x,y, z
    if (x < 0 || x > n - 1)
        x = n / 2;
    if (y < 0 || y > n - 1)
        y = n / 2;
    if (z < 0 || z > n - 1)
        z = n / 2;

    source[(z * n + y) * n + x] = amplitude;

    // Calculate the resulting field with mixed boundary conditions
    double* result = poisson_mixed(n, source, iterations, delta);

    // Print out the middle slice of the cube for validation
    if (debug) {
        printf("--MIDDLE--\n");
    }
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            printf("%0.5f ", result[((n / 2) * n + y) * n + x]);
        }
        printf("\n");
    }

    free(source);
    free(result);

    return EXIT_SUCCESS;
}
