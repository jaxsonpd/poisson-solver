/**
 * @file poisson_iter.c
 * @author Jack Duignan (Jdu80@uclive.ac.nz)
 * @date 2024-09-26
 * @brief Functions for performing poisson iterations
 */


#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h>

#include "utils.h"

#include "poisson_iter.h"

void apply_const_boundary(int N, double* next) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            idx(next, N, 0, j, i) = TOP_BOUNDARY_COND;
            idx(next, N, N - 1, j, i) = BOTTOM_BOUNDARY_COND;
        }
    }
}



void apply_von_neuman_boundary_slice(int N, double* source, double* curr, double* next, float delta, slice3D_t slice_3D) {
    for (int k = slice_3D.k_start; k < slice_3D.k_end; k++) {
        if (slice_3D.j_start == 0 && slice_3D.i_start == 0) {
            idx(next, N, k, 0, 0) = (2 * idx(curr, N, k, 0, 0 + 1)
                + 2 * idx(curr, N, k, 0 + 1, 0)
                + idx(curr, N, k + 1, 0, 0) + idx(curr, N, k - 1, 0, 0)
                - delta * delta * idx(source, N, k, 0, 0)) / 6;
        }

        if (slice_3D.j_end == N && slice_3D.i_end == N) {
            idx(next, N, k, N - 1, N - 1) = (2 * idx(curr, N, k, N - 1, N - 1 - 1)
                + 2 * idx(curr, N, k, N - 1 - 1, N - 1)
                + idx(curr, N, k + 1, N - 1, N - 1) + idx(curr, N, k - 1, N - 1, N - 1)
                - delta * delta * idx(source, N, k, N - 1, N - 1)) / 6;
        }

        if (slice_3D.j_end == N && slice_3D.i_start == 0) {
            idx(next, N, k, N - 1, 0) = (2 * idx(curr, N, k, N - 1, 0 + 1)
                + 2 * idx(curr, N, k, N - 1 - 1, 0)
                + idx(curr, N, k + 1, N - 1, 0) + idx(curr, N, k - 1, N - 1, 0)
                - delta * delta * idx(source, N, k, N - 1, 0)) / 6;
        }

        if (slice_3D.j_start == 0 && slice_3D.i_end == N) {
            idx(next, N, k, 0, N - 1) = (2 * idx(curr, N, k, 0, N - 1 - 1)
                + 2 * idx(curr, N, k, 0 + 1, N - 1)
                + idx(curr, N, k + 1, 0, N - 1) + idx(curr, N, k - 1, 0, N - 1)
                - delta * delta * idx(source, N, k, 0, N - 1)) / 6;
        }
    
        if (slice_3D.i_start == 0) {
            for (int j = slice_3D.j_start + 1; j < slice_3D.j_end - 1; j++) {
                idx(next, N, k, j, 0) = (2 * idx(curr, N, k, j, 0 + 1)
                    + idx(curr, N, k, j + 1, 0) + idx(curr, N, k, j - 1, 0)
                    + idx(curr, N, k + 1, j, 0) + idx(curr, N, k - 1, j, 0)
                    - delta * delta * idx(source, N, k, j, 0)) / 6;
            }
        }

        if (slice_3D.i_end == N) {
            for (int j = slice_3D.j_start + 1; j < slice_3D.j_end - 1; j++) {
                idx(next, N, k, j, N - 1) = (2 * idx(curr, N, k, j, N - 1 - 1)
                    + idx(curr, N, k, j + 1, N - 1) + idx(curr, N, k, j - 1, N - 1)
                    + idx(curr, N, k + 1, j, N - 1) + idx(curr, N, k - 1, j, N - 1)
                    - delta * delta * idx(source, N, k, j, N - 1)) / 6;
            }
        }

        if (slice_3D.j_start == 0) {
            for (int i = slice_3D.i_start + 1; i < slice_3D.i_end - 1; i++) {
                idx(next, N, k, 0, i) = (idx(curr, N, k, 0, i + 1) + idx(curr, N, k, 0, i - 1)
                    + 2 * idx(curr, N, k, 0 + 1, i)
                    + idx(curr, N, k + 1, 0, i) + idx(curr, N, k - 1, 0, i)
                    - delta * delta * idx(source, N, k, 0, i)) / 6;
            }
        }

        if (slice_3D.j_end == N) {
            for (int i = slice_3D.i_start + 1; i < slice_3D.i_end - 1; i++) {
                idx(next, N, k, N - 1, i) = (idx(curr, N, k, N - 1, i + 1) + idx(curr, N, k, N - 1, i - 1)
                    + 2 * idx(curr, N, k, N - 1 - 1, i)
                    + idx(curr, N, k + 1, N - 1, i) + idx(curr, N, k - 1, N - 1, i)
                    - delta * delta * idx(source, N, k, N - 1, i)) / 6;
            }
        }
    }
}

void poisson_iteration_inner_slice(int N, double* source, double* curr, double* next, float delta, slice3D_t slice_3D) {
    int j_start = slice_3D.j_start  != 0 ? slice_3D.j_start : 1;
    int j_end = slice_3D.j_end != N ? slice_3D.j_end : N-1;

    int i_start = slice_3D.i_start  != 0 ? slice_3D.i_start : 1;
    int i_end = slice_3D.i_end != N ? slice_3D.i_end : N-1;
    
    for (int k = slice_3D.k_start; k < slice_3D.k_end; k++) {
        for (int j = j_start; j < j_end; j++) {
            for (int i = i_start; i < i_end; i++) {
                idx(next, N, k, j, i) = (idx(curr, N, k, j, i + 1) + idx(curr, N, k, j, i - 1)
                        + idx(curr, N, k, j + 1, i) + idx(curr, N, k, j - 1, i)
                        + idx(curr, N, k + 1, j, i) + idx(curr, N, k - 1, j, i)
                    - delta * delta * idx(source, N, k, j, i)) / 6;
            }
        }
    }
}

void poisson_iteration_slow(int N, double* source, double* curr, double* next, float delta, slice3D_t slice_3D) {
    for (int k = slice_3D.k_start; k < slice_3D.k_end; k++) {
        for (int j = slice_3D.j_start; j < slice_3D.j_end; j++) {
            for (int i = slice_3D.i_start; i < slice_3D.i_end; i++) {
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

// int intex_SIMD(int N, int z, int y, int x) {
//     // Does the #define idx(array, N, z, y, x) (array[((z)*N + (y)) * N + (x)]) operations
//     // for the SIMD implementation
//     // Apply SIMD with immintrin.h to the following function:
//     // int index = ((z*N) + (y)) * N + (x);


//     return index;
// }

// void poisson_iteration_inner_slice_SIMD(int N, double* source, double* curr, double* next, float delta, slice3D_t slice_3D) {
//     int j_start = slice_3D.j_start  != 0 ? slice_3D.j_start : 1;
//     int j_end = slice_3D.j_end != N ? slice_3D.j_end : N-1;

//     int i_start = slice_3D.i_start  != 0 ? slice_3D.i_start : 1;
//     int i_end = slice_3D.i_end != N ? slice_3D.i_end : N-1;
    

//     // Implements SIMD for iterations over i
//     for (int k = slice_3D.k_start; k < slice_3D.k_end; k++) {
//         for (int j = j_start; j < j_end; j++) {
//             // for (int i = i_start; i < i_end; i+=2) {

//             // }
//             for (int i = i_start; i < i_end; i++) {
//                 next[idx_function(N, k, j, i)] = (curr[idx_function(N, k, j, i + 1)] + curr[idx_function(N, k, j, i - 1)]
//                     + curr[idx_function(N, k, j + 1, i)] + curr[idx_function(N, k, j - 1, i)]
//                     + curr[idx_function(N, k + 1, j, i)] + curr[idx_function(N, k - 1, j, i)]
//                     - delta * delta * source[idx_function(N, k, j, i)]) / 6;
//             }
//             // for (int i = i_start; i < i_end; i++) {
//             //     idx(next, N, k, j, i) = (idx(curr, N, k, j, i + 1) + idx(curr, N, k, j, i - 1)
//             //             + idx(curr, N, k, j + 1, i) + idx(curr, N, k, j - 1, i)
//             //             + idx(curr, N, k + 1, j, i) + idx(curr, N, k - 1, j, i)
//             //         - delta * delta * idx(source, N, k, j, i)) / 6;
//             // }
//         }
//     }
// }

int intex_SIMD(int N, int z, int y, int x) {
    return ((z * N) + y) * N + x;
}

void poisson_iteration_inner_slice_SIMD(int N, double* source, double* curr, double* next, float delta, slice3D_t slice_3D) {
    int j_start = slice_3D.j_start != 0 ? slice_3D.j_start : 1;
    int j_end = slice_3D.j_end != N ? slice_3D.j_end : N - 1;

    int i_start = slice_3D.i_start != 0 ? slice_3D.i_start : 1;
    int i_end = slice_3D.i_end != N ? slice_3D.i_end : N - 1;

    __m256d delta_vec = _mm256_set1_pd(delta * delta); // Convert delta to vector
    __m256d six_vec = _mm256_set1_pd(6.0); // Convert 6 to a vector

    for (int k = slice_3D.k_start; k < slice_3D.k_end; k++) {
        for (int j = j_start; j < j_end; j++) {
            int i;
            // Main SIMD loop (process in blocks of 4)
            for (i = i_start; i <= i_end - 4; i += 4) {
                // Load the current values into vectors
                __m256d curr_vec1 = _mm256_loadu_pd(&curr[intex_SIMD(N, k, j, i + 1)]);
                __m256d curr_vec2 = _mm256_loadu_pd(&curr[intex_SIMD(N, k, j, i - 1)]);
                __m256d curr_vec3 = _mm256_loadu_pd(&curr[intex_SIMD(N, k, j + 1, i)]);
                __m256d curr_vec4 = _mm256_loadu_pd(&curr[intex_SIMD(N, k, j - 1, i)]);
                __m256d curr_vec5 = _mm256_loadu_pd(&curr[intex_SIMD(N, k + 1, j, i)]);
                __m256d curr_vec6 = _mm256_loadu_pd(&curr[intex_SIMD(N, k - 1, j, i)]);
                __m256d source_vec = _mm256_loadu_pd(&source[intex_SIMD(N, k, j, i)]);

                // Perform calculations on vectors
                __m256d sum_vec = _mm256_add_pd(curr_vec1, curr_vec2);
                sum_vec = _mm256_add_pd(sum_vec, curr_vec3);
                sum_vec = _mm256_add_pd(sum_vec, curr_vec4);
                sum_vec = _mm256_add_pd(sum_vec, curr_vec5);
                sum_vec = _mm256_add_pd(sum_vec, curr_vec6);
                sum_vec = _mm256_sub_pd(sum_vec, _mm256_mul_pd(delta_vec, source_vec));
                sum_vec = _mm256_div_pd(sum_vec, six_vec);

                // Store result in 'next'
                _mm256_storeu_pd(&next[intex_SIMD(N, k, j, i)], sum_vec);
            }

            // Scalar cleanup for remaining elements (if any)
            for (; i < i_end; i++) {
                next[intex_SIMD(N, k, j, i)] = (curr[intex_SIMD(N, k, j, i + 1)] +
                                                 curr[intex_SIMD(N, k, j, i - 1)] +
                                                 curr[intex_SIMD(N, k, j + 1, i)] +
                                                 curr[intex_SIMD(N, k, j - 1, i)] +
                                                 curr[intex_SIMD(N, k + 1, j, i)] +
                                                 curr[intex_SIMD(N, k - 1, j, i)]
                    - delta * delta * source[intex_SIMD(N, k, j, i)]) / 6.0;
            }
        }
    }
}


void poisson_iteration_inner_slice_SIMD_half(int N, double* source, double* curr, double* next, float delta, slice3D_t slice_3D) {
    int j_start = slice_3D.j_start != 0 ? slice_3D.j_start : 1;
    int j_end = slice_3D.j_end != N ? slice_3D.j_end : N - 1;

    int i_start = slice_3D.i_start != 0 ? slice_3D.i_start : 1;
    int i_end = slice_3D.i_end != N ? slice_3D.i_end : N - 1;

    __m256d delta_vec = _mm256_set1_pd(delta * delta); // Convert the delta to a vector
    __m256d six_vec = _mm256_set1_pd(6.0); // Convert the 6 to a vectors

    for (int k = slice_3D.k_start; k < slice_3D.k_end; k++) {
        for (int j = j_start; j < j_end; j++) {
            for (int i = i_start; i < i_end; i += 2) {
                // Load the current values into vectors
                __m256d curr_vec1 = _mm256_loadu_pd(&curr[intex_SIMD(N, k, j, i + 1)]);
                __m256d curr_vec2 = _mm256_loadu_pd(&curr[intex_SIMD(N, k, j, i - 1)]);
                __m256d curr_vec3 = _mm256_loadu_pd(&curr[intex_SIMD(N, k, j + 1, i)]);
                __m256d curr_vec4 = _mm256_loadu_pd(&curr[intex_SIMD(N, k, j - 1, i)]);
                __m256d curr_vec5 = _mm256_loadu_pd(&curr[intex_SIMD(N, k + 1, j, i)]);
                __m256d curr_vec6 = _mm256_loadu_pd(&curr[intex_SIMD(N, k - 1, j, i)]);
                __m256d source_vec = _mm256_loadu_pd(&source[intex_SIMD(N, k, j, i)]);

                // Perform the calculations on vectors
                __m256d sum_vec = _mm256_add_pd(curr_vec1, curr_vec2);
                sum_vec = _mm256_add_pd(sum_vec, curr_vec3);
                sum_vec = _mm256_add_pd(sum_vec, curr_vec4);
                sum_vec = _mm256_add_pd(sum_vec, curr_vec5);
                sum_vec = _mm256_add_pd(sum_vec, curr_vec6);
                sum_vec = _mm256_sub_pd(sum_vec, _mm256_mul_pd(delta_vec, source_vec));
                sum_vec = _mm256_div_pd(sum_vec, six_vec);

                _mm256_storeu_pd(&next[intex_SIMD(N, k, j, i)], sum_vec);
            }
        }
    }
}