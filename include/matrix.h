#ifndef MATRIX_H
#define MATRIX_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h> 

// Allocate memory for a flattened 1D matrix
double* allocateFlattenedMatrix(int rows, int cols);

// Free allocated flattened 1D matrix memory
void freeFlattenedMatrix(double* mat);

// Allocate memory for a 2D matrix
double** allocateMatrix(int rows, int cols);

// Free allocated 2D matrix memory
void freeMatrix(double** mat, int rows);

#ifdef __cplusplus
}
#endif

#endif // MATRIX_H