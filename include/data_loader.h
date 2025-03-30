#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "matrix.h"

// --- Function Declarations ---

// Read MNIST dataset images
double** loadMNISTImages(const char* filename, int numImages);

// Read MNIST dataset labels)
double** loadMNISTLabels(const char* filename, int numLabels);

#ifdef __cplusplus
}
#endif

#endif // DATA_LOADER_H