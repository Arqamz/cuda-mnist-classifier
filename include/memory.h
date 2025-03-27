#ifndef MEMORY_H
#define MEMORY_H

#include <stdlib.h>

// Memory management for matrices
double** allocateMatrix(int rows, int cols);
void freeMatrix(double** mat, int rows);

#endif // MEMORY_H