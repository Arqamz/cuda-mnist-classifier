#include "nn.h"

// Main function
int main() {
    printf("MNIST Neural Network\n\n");

    double** train_images = loadMNISTImages("data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    freeNetwork(net);
    return 0;
}
