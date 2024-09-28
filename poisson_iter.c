/**
 * @file poisson_iter.c
 * @author Jack Duignan (Jdu80@uclive.ac.nz)
 * @date 2024-09-26
 * @brief Functions for performing poisson iterations
 */


#include <stdint.h>
#include <stdbool.h>

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

        for (int j = slice_3D.j_start + 1; j < slice_3D.j_end - 1; j++) {
            idx(next, N, k, j, 0) = (2 * idx(curr, N, k, j, 0 + 1)
                + idx(curr, N, k, j + 1, 0) + idx(curr, N, k, j - 1, 0)
                + idx(curr, N, k + 1, j, 0) + idx(curr, N, k - 1, j, 0)
                - delta * delta * idx(source, N, k, j, 0)) / 6;

            idx(next, N, k, j, N - 1) = (2 * idx(curr, N, k, j, N - 1 - 1)
                + idx(curr, N, k, j + 1, N - 1) + idx(curr, N, k, j - 1, N - 1)
                + idx(curr, N, k + 1, j, N - 1) + idx(curr, N, k - 1, j, N - 1)
                - delta * delta * idx(source, N, k, j, N - 1)) / 6;
        }

        for (int i = slice_3D.i_start + 1; i < slice_3D.i_end - 1; i++) {
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