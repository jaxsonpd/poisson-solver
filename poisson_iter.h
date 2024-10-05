/**
 * @file poisson_iter.h
 * @author Jack Duignan (Jdu80@uclive.ac.nz)
 * @date 2024-09-26
 * @brief Functions for conducting possion iterations
 */


#ifndef POISSON_ITER_H
#define POISSON_ITER_H


#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Apply the dirlect boundary conditions to the top and bottom of the
 * cube
 * @param N the length of the cube
 * @param next the array to populate
 *
 */
void apply_const_boundary(int N, double* next);


/**
 * @brief Apply the von neuman boundary condition
 *
 * @param N the length of the sides of the cube
 * @param source the sourcing term
 * @param curr the current state of the calculation
 * @param next the next array to populate
 * @param delta the delta
 * @param slice_3D The slice of data to perform it on
 *
 */
void apply_von_neuman_boundary_slice(int N, double* source, double* curr, double* next, float delta, slice3D_t slice_3D);

/**
 * @brief Perform one iteration of the poisson equation only on the inner squares
 *
 * @param N the size of the array
 * @param source Pointer to the source term
 * @param curr Pointer to the current array
 * @param next Pointer to the next array (to update)
 * @param delta The delta
 * @param slice_3D The slice of data to perform it on
 *
 */
void poisson_iteration_inner_slice(int N, double* source, double* curr, double* next, float delta, slice3D_t slice_3D);

/**
 * @brief Perform one interation of the poisson equation (without constant boundaries)
 *
 * @param N the size of the array
 * @param source Pointer to the source term
 * @param curr Pointer to the current array
 * @param next Pointer to the next array (to update)
 * @param delta The delta
 * @param slice_3D The slice of data to perform it on
 */
void poisson_iteration_slow(int N, double* source, double* curr, double* next, float delta, slice3D_t slice_3D);


#endif // POISSON_ITER_H