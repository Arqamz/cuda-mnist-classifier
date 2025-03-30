#ifndef ACTIVATION_H
#define ACTIVATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h> 

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_utils.cuh"

// --- Kernels ---

// ReLU: Apply ReLU activation
__global__ void relu_kernel(double* x, int size);

// Exponential: Calculate exp(x_i - max_val)
__global__ void exp_kernel(double* x, int size, double max_val);

// Normalize: Normalize array by sum or assign uniform distribution if sum is zero.
__global__ void divide_by_sum_kernel(double* x, int size, double sum);

// --- Wrapper Functions ---

// ReLU: Apply ReLU activation function on GPU
void relu_gpu(double* d_x, int size);

// Softmax: Calculate exp on GPU, sum on CPU, divide on GPU
void softmax_gpu(double* d_x, int size);

#ifdef __cplusplus
}
#endif

#endif // ACTIVATION_H