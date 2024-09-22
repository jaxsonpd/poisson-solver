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


#endif // UTILS_H