/** 
 * @file utils.h
 * @author Jack Duignan (Jdu80@uclive.ac.nz)
 * @date 2024-09-23
 * @brief Various utilities
 */


#ifndef UTILS_H
#define UTILS_H


#include <stdint.h>
#include <stdbool.h>

#define TOP_BOUNDARY_COND -1 // V The top dirlec boundary condition
#define BOTTOM_BOUNDARY_COND 1 // V The bottom dirlec boundary condition

/**
 * @brief index a 3d array
 * @param array the array to index
 * @param N the length of a element
 * @param z the z coord
 * @param y the y coord
 * @param x the x coord
 *optarg
 */
#define idx(array, N, z, y, x) (array[((z)*N + (y)) * N + (x)])

/**
 * @struct Slice3D
 * Store a 3D slice of an array
 */
typedef struct Slice3D {
    int k_start;    ///< The start of the k slice
    int k_end;      ///< The end of the k slice
    int j_start;    ///< The start of the j slice
    int j_end;      ///< The end of the j slice
    int i_start;    ///< The start of the i slice
    int i_end;      ///< The end of the i slice
} slice3D_t;  

/** 
 * @brief Copy a slice of three dimentional array
 * @param N the length of the cube
 * @param curr the current array
 * @param next the next array
 * @param slice_3D the slice to copy
 * 
 */
void memcopy_3D(int N, double *curr, double *next, slice3D_t slice_3D);


#endif // UTILS_H