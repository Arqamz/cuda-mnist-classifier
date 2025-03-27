#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>

#include "timer.h"
#include "config.h"
#include "activation.h"
#include "memory.h"

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

#endif // NEURAL_NETWORK_H
