#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "memory.h"

// MNIST dataset loading
double** loadMNISTImages(const char* filename, int numImages);
double** loadMNISTLabels(const char* filename, int numLabels);

#endif // DATA_LOADER_H