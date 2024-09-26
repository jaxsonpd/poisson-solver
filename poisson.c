#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>

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

// Global flag
// Set to true when operating in debug mode to enable verbose logging
static bool debug = false;

// Statics
const double top_boundary_cond = -1; // V The top dirlec boundary condition
const double bottom_boundary_cond = 1; // V The bottom dirlec boundary condition

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

    pthread_t worker_threads[threads];
    workerThread_t thread_info[threads];
    pthread_barrier_t barrier;

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

    pthread_barrier_init(&barrier, NULL, threads);

    

    int thickness = ceil((N-2)/(float)threads);

    // Launch each of the new worker threads
    for (int i = 0; i < threads; i++) {
        
        // Fill in the arguments to the worker
        thread_info[i].thread_id = i;
        thread_info[i].N = N;
        thread_info[i].source = source;
        thread_info[i].curr = curr;
        thread_info[i].next = next;
        thread_info[i].delta = delta;
        thread_info[i].iterations = iterations;
        // thread_info[i].k_start = 1 + ((N * i) / threads) ; // This is probably wrong
        // thread_info[i].k_end = -1 + (N * (i+1)) / threads; // This too
        thread_info[i].slice_3D.k_start = i*thickness+1;
        int k_end = (i+1)*thickness+1;
        thread_info[i].slice_3D.k_end = k_end > (N-1) ? N-1 : k_end;
        // printf("Thread: %d, k_start %d, k_end %d\n", i, thread_info[i].k_start, thread_info[i].k_end);

        thread_info[i].slice_3D.j_start = 0;
        thread_info[i].slice_3D.j_end = N;
        thread_info[i].slice_3D.i_start = 0;
        thread_info[i].slice_3D.i_end = N;
        thread_info[i].barrier = &barrier;

        // Create the worker thread
        if (pthread_create (&worker_threads[i], NULL, &worker_thread, &thread_info[i]) != 0)
        {
            fprintf (stderr, "Error creating worker thread!\n");
        }
    }

    // Wait for all the threads to finish using join ()
    for (int i = 0; i < threads; i++)
    {
        pthread_join (worker_threads[i], NULL);
    }

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
