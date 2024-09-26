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