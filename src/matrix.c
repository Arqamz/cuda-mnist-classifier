#include "matrix.h"

// --- Flattened 1D Matrix Allocation ---

double* allocateFlattenedMatrix(int rows, int cols) {
    double* mat = (double*)malloc(rows * cols * sizeof(double));
    if (!mat) {
        fprintf(stderr, "Error: Failed to allocate flattened matrix memory (%d x %d)\n", rows, cols);
        exit(EXIT_FAILURE);
    }
    return mat;
}

void freeFlattenedMatrix(double* mat) {
    free(mat);
}

// --- Standard 2D Matrix Allocation ---

double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}