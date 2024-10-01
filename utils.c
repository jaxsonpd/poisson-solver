/** 
 * @file utils.c
 * @author Jack Duignan (Jdu80@uclive.ac.nz)
 * @date 2024-09-26
 * @brief Various utility functions
 */


#include <stdint.h>
#include <stdbool.h>

#include "utils.h"

void memcopy_3D(int N, double *curr, double *next, slice3D_t slice_3D) {
    for (int k = slice_3D.k_start; k < slice_3D.k_end; k++) {
            for (int j = slice_3D.j_start; j < slice_3D.j_end; j++) {
                for (int i = slice_3D.i_start; i < slice_3D.i_end; i++) {
                    idx(curr, N, k, j, i) = idx(next, N, k, j, i);
                }
            }
        }
}