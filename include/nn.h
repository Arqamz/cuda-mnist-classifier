#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Constants
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9

// Uncomment to enable verbose mode
// #define VERBOSE

// Macro for verbose mode print statements
#ifdef VERBOSE
    #define VERBOSE_PRINT(fmt, ...) printf(fmt, __VA_ARGS__)
#else
    #define VERBOSE_PRINT(fmt, ...)  // Do nothing when VERBOSE is not defined
#endif

// Timer function
double get_time(clock_t start);

// Memory management for matrices
double** allocateMatrix(int rows, int cols);
void freeMatrix(double** mat, int rows);

// Activation functions
void relu(double* x, int size);
void softmax(double* x, int size);

// Neural network structure
typedef struct {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
} NeuralNetwork;

// Neural network operations
NeuralNetwork* createNetwork();
void forward(NeuralNetwork* net, double* input, double* hidden, double* output);
void backward(NeuralNetwork* net, double* input, double* hidden, double* output, double* target);
void train(NeuralNetwork* net, double** images, double** labels, int numImages);
void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages);
void freeNetwork(NeuralNetwork* net);

// MNIST dataset loading
double** loadMNISTImages(const char* filename, int numImages);
double** loadMNISTLabels(const char* filename, int numLabels);

#endif // NN_H
