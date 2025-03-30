#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.cuh"

#include "data_loader.h"
#include "matrix.h"
#include "neural_network.cuh"

// Main function
int main() {
    printf("MNIST Neural Network (CUDA Version)\n\n");

    // --- 1. Load Data (CPU) ---
    printf("Loading MNIST data...\n");
    double** train_images_orig = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
    double** train_labels_orig = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);
    double** test_images_orig = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);
    double** test_labels_orig = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);
    printf("Data loading complete.\n\n");

    // --- 2. Initialize Network (CPU Host Memory) ---
    printf("Creating network structure on host...\n");
    NeuralNetwork* net = createNetwork();

    // --- 3. Allocate GPU Memory and Upload Initial Network ---
    printf("Allocating device memory and uploading network to GPU...\n");
    allocateAndUploadNetwork(net);

    // --- 4. Train the Network (using GPU kernels) ---
    // The train_gpu function handles copying individual images/labels per iteration
    printf("Starting training phase...\n");
    train_gpu(net, train_images_orig, train_labels_orig, 60000);
    printf("Training complete.\n\n");

    // --- 5. Evaluate the Network (using GPU forward pass) ---
    printf("Starting evaluation phase...\n");
    evaluate_gpu(net, test_images_orig, test_labels_orig, 10000);
    printf("Evaluation complete.\n\n");

    // --- 6. Cleanup ---
    printf("Cleaning up...\n");
    // Free GPU memory
    freeNetworkDevice(net);
    // Free CPU network memory
    freeNetworkHost(net); // Frees net struct itself too

    // Free original MNIST data loaded on CPU
    printf("Freeing MNIST host data...\n");
    freeMatrix(train_images_orig, 60000);
    freeMatrix(train_labels_orig, 60000);
    freeMatrix(test_images_orig, 10000);
    freeMatrix(test_labels_orig, 10000);

    printf("\nExecution finished.\n");
    return 0;
}