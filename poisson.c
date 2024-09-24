#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

#include "utils.h"

#include "worker_thread.h"


/**
 * poisson.c
 * Implementation of a Poisson solver with Dirichlet boundary conditions.
 *
 * This template handles the basic program launch, argument parsing, and memory
 * allocation required to implement the solver *at its most basic level*. You
 * will likely need to allocate more memory, add threading support, account for
 * cache locality, etc...
 *
 * BUILDING:
 * gcc -o poisson poisson.c -lpthread
 *
 * [note: linking pthread isn't strictly needed until you add your
 *        multithreading code]
 *
 * TODO:
 * 1 - Read through this example, understand what it does and what it gives you
 *     to work with.
 * 2 - Implement the basic algorithm and get a correct output.
 * 3 - Add a timer to track how long your execution takes.
 * 4 - Profile your solution and identify weaknesses.
 * 5 - Improve it!
 * 6 - Remember that this is now *your* code and *you* should modify it however
 *     needed to solve the assignment.
 *
 * See the lab notes for a guide on profiling and an introduction to
 * multithreading (see also threads.c which is reference by the lab notes).
 */

extern char* optarg;

pthread_t single_thread;

// Global flag
// Set to true when operating in debug mode to enable verbose logging
static bool debug = false;

// Statics
const double top_boundary_cond = -1; // V The top dirlec boundary condition
const double bottom_boundary_cond = 1; // V The bottom dirlec boundary condition

/**
 * @brief Apply the dirlect boundary conditions to the top and bottom of the
 * cube
 * @param N the length of the cube
 * @param next the array to populate
 *
 */
void apply_const_boundary(int N, double* next) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            idx(next, N, 0, j, i) = TOP_BOUNDARY_COND;
            idx(next, N, N - 1, j, i) = BOTTOM_BOUNDARY_COND;
        }
    }
}

/**
 * @brief Apply the von neuman boundary condition
 *
 * @param N the length of the sides of the cube
 * @param source the sourcing term
 * @param curr the current state of the calculation
 * @param next the next array to populate
 * @param delta the delta
 *
 */
void apply_von_neuman_boundary(int N, double* source, double* curr, double* next, float delta) {
    for (int k = 1; k < N - 1; k++) {
        idx(next, N, k, 0, 0) = (2 * idx(curr, N, k, 0, 0 + 1)
            + 2 * idx(curr, N, k, 0 + 1, 0)
            + idx(curr, N, k + 1, 0, 0) + idx(curr, N, k - 1, 0, 0)
            - delta * delta * idx(source, N, k, 0, 0)) / 6;

        idx(next, N, k, N - 1, N - 1) = (2 * idx(curr, N, k, N - 1, N - 1 - 1)
            + 2 * idx(curr, N, k, N - 1 - 1, N - 1)
            + idx(curr, N, k + 1, N - 1, N - 1) + idx(curr, N, k - 1, N - 1, N - 1)
            - delta * delta * idx(source, N, k, N - 1, N - 1)) / 6;

        idx(next, N, k, N - 1, 0) = (2 * idx(curr, N, k, N - 1, 0 + 1)
            + 2 * idx(curr, N, k, N - 1 - 1, 0)
            + idx(curr, N, k + 1, N - 1, 0) + idx(curr, N, k - 1, N - 1, 0)
            - delta * delta * idx(source, N, k, N - 1, 0)) / 6;

        idx(next, N, k, 0, N - 1) = (2 * idx(curr, N, k, 0, N - 1 - 1)
            + 2 * idx(curr, N, k, 0 + 1, N - 1)
            + idx(curr, N, k + 1, 0, N - 1) + idx(curr, N, k - 1, 0, N - 1)
            - delta * delta * idx(source, N, k, 0, N - 1)) / 6;

        for (int j = 1; j < N - 1; j++) {
            idx(next, N, k, j, 0) = (2 * idx(curr, N, k, j, 0 + 1)
                + idx(curr, N, k, j + 1, 0) + idx(curr, N, k, j - 1, 0)
                + idx(curr, N, k + 1, j, 0) + idx(curr, N, k - 1, j, 0)
                - delta * delta * idx(source, N, k, j, 0)) / 6;

            idx(next, N, k, j, N - 1) = (2 * idx(curr, N, k, j, N - 1 - 1)
                + idx(curr, N, k, j + 1, N - 1) + idx(curr, N, k, j - 1, N - 1)
                + idx(curr, N, k + 1, j, N - 1) + idx(curr, N, k - 1, j, N - 1)
                - delta * delta * idx(source, N, k, j, N - 1)) / 6;
        }

        for (int i = 1; i < N - 1; i++) {
            idx(next, N, k, 0, i) = (idx(curr, N, k, 0, i + 1) + idx(curr, N, k, 0, i - 1)
                + 2 * idx(curr, N, k, 0 + 1, i)
                + idx(curr, N, k + 1, 0, i) + idx(curr, N, k - 1, 0, i)
                - delta * delta * idx(source, N, k, 0, i)) / 6;

            idx(next, N, k, N - 1, i) = (idx(curr, N, k, N - 1, i + 1) + idx(curr, N, k, N - 1, i - 1)
                + 2 * idx(curr, N, k, N - 1 - 1, i)
                + idx(curr, N, k + 1, N - 1, i) + idx(curr, N, k - 1, N - 1, i)
                - delta * delta * idx(source, N, k, N - 1, i)) / 6;
        }
    }
}

/**
 * @brief Perform one iteration of the poisson equation with optimised loops
 *
 * @param N the size of the array
 * @param source Pointer to the source term
 * @param curr Pointer to the current array
 * @param next Pointer to the next array (to update)
 * @param delta The delta
 *
 */
void poisson_iteration_faster(int N, double* source, double* curr, double* next, float delta) {
    apply_const_boundary(N, next);

    apply_von_neuman_boundary(N, source, curr, next, delta);

    for (int k = 1; k < N - 1; k++) {
        for (int j = 1; j < N - 1; j++) {
            for (int i = 1; i < N - 1; i++) {
                idx(next, N, k, j, i) = (idx(curr, N, k, j, i + 1) + idx(curr, N, k, j, i - 1)
                    + idx(curr, N, k, j + 1, i) + idx(curr, N, k, j - 1, i)
                    + idx(curr, N, k + 1, j, i) + idx(curr, N, k - 1, j, i)
                    - delta * delta * idx(source, N, k, j, i)) / 6;
            }
        }
    }
}

/**
 * @brief Perform one interation of the poisson equation
 *
 * @param N the size of the array
 * @param source Pointer to the source term
 * @param curr Pointer to the current array
 * @param next Pointer to the next array (to update)
 * @param delta The delta
 */
void poisson_iteration_slow(int N, double* source, double* curr, double* next, float delta) {
    // Apply constant boundary
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            idx(next, N, 0, j, i) = TOP_BOUNDARY_COND;
            idx(next, N, N - 1, j, i) = BOTTOM_BOUNDARY_COND;
        }
    }

    // Apply Neumann Boundary


    for (int k = 1; k < N - 1; k++) {
        for (int j = 0; j < N; j++) {
            for (int i = 0; i < N; i++) {
                if (i == 0 && j == 0) {
                    idx(next, N, k, j, i) = (2 * idx(curr, N, k, j, i + 1)
                        + 2 * idx(curr, N, k, j + 1, i)
                        + idx(curr, N, k + 1, j, i) + idx(curr, N, k - 1, j, i)
                        - delta * delta * idx(source, N, k, j, i)) / 6;
                } else if (i == N - 1 && j == N - 1) {
                    idx(next, N, k, j, i) = (2 * idx(curr, N, k, j, i - 1)
                        + 2 * idx(curr, N, k, j - 1, i)
                        + idx(curr, N, k + 1, j, i) + idx(curr, N, k - 1, j, i)
                        - delta * delta * idx(source, N, k, j, i)) / 6;
                } else if (i == 0 && j == N - 1) {
                    idx(next, N, k, j, i) = (2 * idx(curr, N, k, j, i + 1)
                        + 2 * idx(curr, N, k, j - 1, i)
                        + idx(curr, N, k + 1, j, i) + idx(curr, N, k - 1, j, i)
                        - delta * delta * idx(source, N, k, j, i)) / 6;
                } else if (i == N - 1 && j == 0) {
                    idx(next, N, k, j, i) = (2 * idx(curr, N, k, j, i - 1)
                        + 2 * idx(curr, N, k, j + 1, i)
                        + idx(curr, N, k + 1, j, i) + idx(curr, N, k - 1, j, i)
                        - delta * delta * idx(source, N, k, j, i)) / 6;
                } else if (i == 0) {
                    idx(next, N, k, j, i) = (2 * idx(curr, N, k, j, i + 1)
                        + idx(curr, N, k, j + 1, i) + idx(curr, N, k, j - 1, i)
                        + idx(curr, N, k + 1, j, i) + idx(curr, N, k - 1, j, i)
                        - delta * delta * idx(source, N, k, j, i)) / 6;
                } else if (i == N - 1) {
                    idx(next, N, k, j, i) = (2 * idx(curr, N, k, j, i - 1)
                        + idx(curr, N, k, j + 1, i) + idx(curr, N, k, j - 1, i)
                        + idx(curr, N, k + 1, j, i) + idx(curr, N, k - 1, j, i)
                        - delta * delta * idx(source, N, k, j, i)) / 6;
                } else if (j == 0) {
                    idx(next, N, k, j, i) = (idx(curr, N, k, j, i + 1) + idx(curr, N, k, j, i - 1)
                        + 2 * idx(curr, N, k, j + 1, i)
                        + idx(curr, N, k + 1, j, i) + idx(curr, N, k - 1, j, i)
                        - delta * delta * idx(source, N, k, j, i)) / 6;
                } else if (j == N - 1) {
                    idx(next, N, k, j, i) = (idx(curr, N, k, j, i + 1) + idx(curr, N, k, j, i - 1)
                        + 2 * idx(curr, N, k, j - 1, i)
                        + idx(curr, N, k + 1, j, i) + idx(curr, N, k - 1, j, i)
                        - delta * delta * idx(source, N, k, j, i)) / 6;

                } else {
                    idx(next, N, k, j, i) = (idx(curr, N, k, j, i + 1) + idx(curr, N, k, j, i - 1)
                        + idx(curr, N, k, j + 1, i) + idx(curr, N, k, j - 1, i)
                        + idx(curr, N, k + 1, j, i) + idx(curr, N, k - 1, j, i)
                        - delta * delta * idx(source, N, k, j, i)) / 6;;
                }
            }
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
double* poisson_mixed(int N, double* source, int iterations, int threads, float delta) {
    if (debug) {
        printf("Starting solver with:\n"
            "n = %i\n"
            "iterations = %i\n"
            "threads = %i\n"
            "delta = %f\n",
            N, iterations, threads, delta);
    }

    // Allocate some buffers to calculate the solution in
    double* curr = (double*)calloc(N * N * N, sizeof(double));
    double* next = (double*)calloc(N * N * N, sizeof(double));
    // Ensure we haven't run out of memory
    if (curr == NULL || next == NULL) {
        fprintf(stderr, "Error: ran out of memory when trying to allocate %i sized cube\n", N);
        exit(EXIT_FAILURE);
    }

    // Apply constant boundary
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            idx(next, N, 0, j, i) = TOP_BOUNDARY_COND;
            idx(next, N, N - 1, j, i) = BOTTOM_BOUNDARY_COND;
        }
    }

    workerThread_t worker_info = {
        .thread_id = 0,

        .N = N,
        .source = source,
        .curr = curr,
        .next = next,
        .delta = delta,
        .iterations = iterations,
        .k_start = 1,
        .k_end = N-1,
        .j_start = 0,
        .j_end = N,
        .i_start = 0,
        .i_end = N
    };

    pthread_create(&single_thread, NULL, worker_thread, &worker_info);

    pthread_join(single_thread, NULL);





    // Free one of the buffers and return the correct answer in the other.
    // The caller is now responsible for free'ing the returned pointer.
    free(next);

    if (debug) {
        printf("Finished solving.\n");
    }

    return (double*)curr;
}



int main(int argc, char** argv) {
    // Default settings for solver
    int iterations = 10;
    int n = 5;
    int threads = 1;
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
    double* result = poisson_mixed(n, source, iterations, threads, delta);

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
