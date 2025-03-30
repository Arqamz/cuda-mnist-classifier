#include "neural_network.cuh"

// Initialize neural network
NeuralNetwork* createNetwork() {
    NeuralNetwork* net = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (!net) {
        fprintf(stderr, "Failed to allocate NeuralNetwork struct\n");
        exit(EXIT_FAILURE);
    }

    // Host memory allocation
    net->W1 = allocateFlattenedMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateFlattenedMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    if (!net->b1 || !net->b2) {
        fprintf(stderr, "Failed to allocate bias vectors\n");
        // Clean up previously allocated memory
        freeFlattenedMatrix(net->W1);
        freeFlattenedMatrix(net->W2);
        free(net->b1);
        free(net->b2);
        free(net);
        exit(EXIT_FAILURE);
    }

    // Initialize weights with small random values
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i * INPUT_SIZE + j] = ((double)rand() / RAND_MAX) * 0.1 - 0.05; // Small range around 0

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i * HIDDEN_SIZE + j] = ((double)rand() / RAND_MAX) * 0.1 - 0.05; // Small range around 0

    // Set device pointers to NULL
    net->d_W1 = NULL;
    net->d_W2 = NULL;
    net->d_b1 = NULL;
    net->d_b2 = NULL;
    net->d_input = NULL;
    net->d_hidden = NULL;
    net->d_output = NULL;
    net->d_target = NULL;
    net->d_d_output = NULL;
    net->d_d_hidden = NULL;
    net->d_temp = NULL;

    printf("Host network allocated and initialized.\n");
    return net;
}

// Allocate GPU memory and copy initial weights/biases
void allocateAndUploadNetwork(NeuralNetwork* net) {
    // Weights and biases
    CHECK_CUDA(cudaMalloc((void**)&net->d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&net->d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&net->d_b1, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&net->d_b2, OUTPUT_SIZE * sizeof(double)));

    // Intermediate buffers (activations, gradients)
    CHECK_CUDA(cudaMalloc((void**)&net->d_input, INPUT_SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&net->d_hidden, HIDDEN_SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&net->d_output, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&net->d_target, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&net->d_d_output, OUTPUT_SIZE * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&net->d_d_hidden, HIDDEN_SIZE * sizeof(double)));

    // Temporary buffer
    CHECK_CUDA(cudaMalloc((void**)&net->d_temp, HIDDEN_SIZE * sizeof(double)));

    // Copy initial weights and biases from Host -> Device
    CHECK_CUDA(cudaMemcpy(net->d_W1, net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_W2, net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_b1, net->b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(net->d_b2, net->b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));

    printf("Device memory allocated and network uploaded to GPU.\n");
}

// --- Kernels for Forward and Backward Pass ---

// Computes y = W*x + b (Matrix-Vector Multiplication + Bias)
// W is (rows x cols), x is (cols x 1), b is (rows x 1), y is (rows x 1)
__global__ void matrix_vector_mul_add_bias_kernel(const double* W, const double* x, const double* b, double* y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows) {
        double sum = 0.0;
        // Calculate dot product W[row, :] * x
        for (int j = 0; j < cols; ++j) {
            sum += W[row * cols + j] * x[j];
        }
        // Add bias
        y[row] = sum + b[row];
    }
}

// Computes d_output = output - target (Element-wise)
__global__ void compute_output_gradient_kernel(const double* output, const double* target, double* d_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_output[idx] = output[idx] - target[idx];
    }
}

// Computes temp = W^T * delta (Matrix Transpose-Vector Multiplication)
// W is (rows x cols), delta is (rows x 1), temp is (cols x 1) 
// W^T is (cols x rows)
__global__ void matrix_transpose_vector_mul_kernel(const double* W, const double* delta, double* temp, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Index corresponds to the output vector (temp)

    if (col < cols) {
        double sum = 0.0;
        // Calculate dot product W[:, col]^T * delta
        for (int i = 0; i < rows; ++i) {
            // Access W column-wise (transpose): W[i][col] == W[i * cols + col]
            sum += W[i * cols + col] * delta[i];
        }
        temp[col] = sum;
    }
}

// Computes d_hidden = temp * (hidden > 0) (Element-wise product with ReLU derivative)
// temp = W2^T * d_output
__global__ void compute_hidden_gradient_relu_kernel(const double* temp, const double* hidden, double* d_hidden, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_hidden[idx] = temp[idx] * (hidden[idx] > 0.0 ? 1.0 : 0.0);
    }
}

// Updates weights W -= learning_rate * delta * x^T (Outer product update)
// W is (rows x cols), delta is (rows x 1), x is (cols x 1)
// Update is delta * x^T which is (rows x cols)
__global__ void update_weights_kernel(double* W, const double* delta, const double* x, double learning_rate, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        int W_idx = row * cols + col;
        W[W_idx] -= learning_rate * delta[row] * x[col];
    }
}

// Updates biases b -= learning_rate * delta (Element-wise)
__global__ void update_biases_kernel(double* b, const double* delta, double learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idx] -= learning_rate * delta[idx];
    }
}

// --- GPU Wrapper Functions ---

void forward_gpu(NeuralNetwork* net, double* d_input, double* d_hidden, double* d_output) {
    int threadsPerBlock = 256;
    int blocksPerGrid;

    // Layer 1: hidden_pre_activation = W1 * input + b1
    blocksPerGrid = (HIDDEN_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    matrix_vector_mul_add_bias_kernel<<<blocksPerGrid, threadsPerBlock>>>(net->d_W1, d_input, net->d_b1, d_hidden, HIDDEN_SIZE, INPUT_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // Activation 1: hidden = relu(hidden_pre_activation)
    relu_gpu(d_hidden, HIDDEN_SIZE); // Uses its own launch config

    // Layer 2: output_pre_activation = W2 * hidden + b2
    blocksPerGrid = (OUTPUT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    matrix_vector_mul_add_bias_kernel<<<blocksPerGrid, threadsPerBlock>>>(net->d_W2, d_hidden, net->d_b2, d_output, OUTPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // Activation 2: output = softmax(output_pre_activation)
    softmax_gpu(d_output, OUTPUT_SIZE); // Uses its own launch config and H->D copies

    CHECK_CUDA(cudaDeviceSynchronize()); // Ensure all forward kernels complete
}

void backward_gpu(NeuralNetwork* net, double* d_input, double* d_hidden, double* d_output, double* d_target) {
    int threadsPerBlock = 256;
    int blocksPerGrid;

    // 1. Compute output layer gradient: d_output = output - target
    blocksPerGrid = (OUTPUT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    compute_output_gradient_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_target, net->d_d_output, OUTPUT_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // 2. Compute hidden layer gradient: d_hidden = (W2^T * d_output) * relu'(hidden_pre_activation)
    //    2a: temp = W2^T * d_output
    blocksPerGrid = (HIDDEN_SIZE + threadsPerBlock - 1) / threadsPerBlock; // Output size is HIDDEN_SIZE
    matrix_transpose_vector_mul_kernel<<<blocksPerGrid, threadsPerBlock>>>(net->d_W2, net->d_d_output, net->d_temp, OUTPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());
    //    2b: d_hidden = temp * (hidden > 0) 
    blocksPerGrid = (HIDDEN_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    compute_hidden_gradient_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(net->d_temp, d_hidden, net->d_d_hidden, HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // 3. Update weights W2: W2 -= LR * d_output * hidden^T
    dim3 threads(16, 16);
    dim3 grid( (OUTPUT_SIZE + threads.x - 1) / threads.x,
               (HIDDEN_SIZE + threads.y - 1) / threads.y );
    update_weights_kernel<<<grid, threads>>>(net->d_W2, net->d_d_output, d_hidden, LEARNING_RATE, OUTPUT_SIZE, HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // 4. Update biases b2: b2 -= LR * d_output
    blocksPerGrid = (OUTPUT_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    update_biases_kernel<<<blocksPerGrid, threadsPerBlock>>>(net->d_b2, net->d_d_output, LEARNING_RATE, OUTPUT_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // 5. Update weights W1: W1 -= LR * d_hidden * input^T
    grid.x = (HIDDEN_SIZE + threads.x - 1) / threads.x;
    grid.y = (INPUT_SIZE + threads.y - 1) / threads.y;
    update_weights_kernel<<<grid, threads>>>(net->d_W1, net->d_d_hidden, d_input, LEARNING_RATE, HIDDEN_SIZE, INPUT_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // 6. Update biases b1: b1 -= LR * d_hidden
    blocksPerGrid = (HIDDEN_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    update_biases_kernel<<<blocksPerGrid, threadsPerBlock>>>(net->d_b1, net->d_d_hidden, LEARNING_RATE, HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize()); // Ensure all backward kernels complete
}

void train_gpu(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();

    // Allocate host buffer for retrieving results (output layer)
    double* h_output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
     if (!h_output) {
        fprintf(stderr, "Failed to allocate host memory for output buffer\n");
        exit(EXIT_FAILURE);
    }

    printf("\nStarting GPU Training...\n");
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double total_loss = 0.0;
        int correct_predictions = 0;

        for (int i = 0; i < numImages; i++) {
            // --- Data Transfer: Host -> Device for current image/label ---
            CHECK_CUDA(cudaMemcpy(net->d_input, images[i], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(net->d_target, labels[i], OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));

            // --- Forward Pass ---
            // Device pointers for input, hidden buffer, output buffer
            forward_gpu(net, net->d_input, net->d_hidden, net->d_output);

            // --- Backward Pass ---
            // Device pointers for input, hidden buffer, output buffer, target label
            backward_gpu(net, net->d_input, net->d_hidden, net->d_output, net->d_target);

            // --- Calculate Loss & Accuracy (CPU) ---
            // Copy output back from Device -> Host
            CHECK_CUDA(cudaMemcpy(h_output, net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));

            // Compute loss & accuracy
            double current_loss = 0.0;
            int predicted_label_idx = 0;
            int actual_label_idx = 0;

            for (int k = 0; k < OUTPUT_SIZE; k++) {
                current_loss -= labels[i][k] * log(fmax(h_output[k], 1e-9)); // Cross-Entropy Loss
                if (h_output[k] > h_output[predicted_label_idx]) {
                    predicted_label_idx = k; // Find predicted label
                }
                if (labels[i][k] > labels[i][actual_label_idx]) {
                    actual_label_idx = k; // Find actual label
                }
            }
            total_loss += current_loss;

            // Check if prediction is correct
            if (predicted_label_idx == actual_label_idx) {
                correct_predictions++;
            }

            // Less prints
            if ((i + 1) % 10000 == 0) {
                 printf("  Epoch %d, Image %d/%d - Avg Loss: %.4f - Current Accuracy: %.2f%%\n",
                       epoch + 1, i + 1, numImages, total_loss / (i + 1),
                       (double)correct_predictions / (i + 1) * 100.0);
            }
        } // end image loop

        double epoch_time = get_time(epoch_start);
        double avg_loss = total_loss / numImages;
        double accuracy = (double)correct_predictions / numImages * 100.0;

        printf("Epoch %d - Avg Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, avg_loss, accuracy, epoch_time);

    } // end epoch loop

    printf("Total training time: %.3fs\n", get_time(total_start));
    free(h_output); // Free host output
}

// Evaluate accuracy on test data (GPU)
void evaluate_gpu(NeuralNetwork* net, double** images, double** labels, int numImages) {
    printf("\nEvaluating on Test Set (GPU Forward Pass)...\n");
    int correct = 0;

    // Host buffer for retrieving results
    double* h_output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    
    if (!h_output) {
        fprintf(stderr, "Failed to allocate host memory for evaluation output buffer\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numImages; i++) {
        // Copy input image to device
        CHECK_CUDA(cudaMemcpy(net->d_input, images[i], INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice));

        // Perform forward pass (GPU)
        forward_gpu(net, net->d_input, net->d_hidden, net->d_output);

        // Copy output back from Device -> Host
        CHECK_CUDA(cudaMemcpy(h_output, net->d_output, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost));

        // Compare guess with actual label
        int pred_idx = 0;
        int actual_idx = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (h_output[j] > h_output[pred_idx]) pred_idx = j;
            if (labels[i][j] > labels[i][actual_idx]) actual_idx = j;
        }
        if (pred_idx == actual_idx) correct++;
    }

    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100.0);
    free(h_output);
}

// Free network host memory
void freeNetworkHost(NeuralNetwork* net) {
    if (!net) return;
    freeFlattenedMatrix(net->W1);
    freeFlattenedMatrix(net->W2);
    free(net->b1);
    free(net->b2);
    free(net);
    printf("Host network memory freed.\n");
}

// Free network device memory
void freeNetworkDevice(NeuralNetwork* net) {
    if (!net) return;
    // errors during cleanup might mask earlier issues therefore no cudaCheck here
    if (net->d_W1) cudaFree(net->d_W1);
    if (net->d_W2) cudaFree(net->d_W2);
    if (net->d_b1) cudaFree(net->d_b1);
    if (net->d_b2) cudaFree(net->d_b2);
    if (net->d_input) cudaFree(net->d_input);
    if (net->d_hidden) cudaFree(net->d_hidden);
    if (net->d_output) cudaFree(net->d_output);
    if (net->d_target) cudaFree(net->d_target);
    if (net->d_d_output) cudaFree(net->d_d_output);
    if (net->d_d_hidden) cudaFree(net->d_d_hidden);
    if (net->d_temp) cudaFree(net->d_temp);

    // Set pointers to NULL after freeing
    net->d_W1 = NULL;
    net->d_W2 = NULL;
    net->d_b1 = NULL;
    net->d_b2 = NULL;
    net->d_input = NULL;
    net->d_hidden = NULL;
    net->d_output = NULL;
    net->d_target = NULL;
    net->d_d_output = NULL;
    net->d_d_hidden = NULL;
    net->d_temp = NULL;

    printf("Device network memory freed.\n");
}