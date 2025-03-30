#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_utils.cuh"

#include "config.h"
#include "data_loader.h"
#include "matrix.h"
#include "timer.h"
#include "activation.cuh"

// --- Neural Network structure ---
typedef struct {
    // Host (CPU) memory pointers
    double* W1;         // Flattened Weight matrix layer 1 (Hidden x Input)
    double* W2;         // Flattened Weight matrix layer 2 (Output x Hidden)
    double* b1;         // Bias vector layer 1
    double* b2;         // Bias vector layer 2

    // Device (GPU) memory pointers
    double* d_W1;
    double* d_W2;
    double* d_b1;
    double* d_b2;

    // Pointers for intermediate activations/gradients on GPU (allocated per training run or per image)
    double* d_input;
    double* d_hidden;
    double* d_output;
    double* d_target;
    double* d_d_output; // Gradient of loss w.r.t output layer pre-activation (output - target)
    double* d_d_hidden; // Gradient of loss w.r.t hidden layer pre-activation
    double* d_temp;     // Temporary buffer

} NeuralNetwork;

// --- Kernels ---

// Computes y = W*x + b (Matrix-Vector Multiplication + Bias)
__global__ void matrix_vector_mul_add_bias_kernel(const double* W, const double* x, const double* b, double* y, int rows, int cols);

// Computes d_output = output - target (Element-wise)
__global__ void compute_output_gradient_kernel(const double* output, const double* target, double* d_output, int size);

// Computes temp = W^T * delta (Matrix Transpose-Vector Multiplication)
__global__ void matrix_transpose_vector_mul_kernel(const double* W, const double* delta, double* temp, int rows, int cols);

// Computes d_hidden = temp * (hidden > 0) (Element-wise product with ReLU derivative)
__global__ void compute_hidden_gradient_relu_kernel(const double* temp, const double* hidden, double* d_hidden, int size);

// Updates weights W -= learning_rate * delta * x^T (Outer product update)
__global__ void update_weights_kernel(double* W, const double* delta, const double* x, double learning_rate, int rows, int cols);

// Updates biases b -= learning_rate * delta (Element-wise)
__global__ void update_biases_kernel(double* b, const double* delta, double learning_rate, int size);

// --- Wrapper functions -- 

// Forward pass
void forward_gpu(NeuralNetwork* net, double* d_input, double* d_hidden, double* d_output);

// Backpropagation
void backward_gpu(NeuralNetwork* net, double* d_input, double* d_hidden, double* d_output, double* d_target);

// Train network
void train_gpu(NeuralNetwork* net, double** images, double** labels, int numImages);

// Evaluate accuracy on test data
void evaluate_gpu(NeuralNetwork* net, double** images, double** labels, int numImages);

// --- Utility/Helper functions ---

// Initialize neural network (allocates HOST memory)
NeuralNetwork* createNetwork();

// Allocate GPU memory and copy initial weights/biases
void allocateAndUploadNetwork(NeuralNetwork* net);

// Free network host memory
void freeNetworkHost(NeuralNetwork* net);

// Free network device memory
void freeNetworkDevice(NeuralNetwork* net);

#ifdef __cplusplus
}
#endif

#endif // NEURAL_NETWORK_H