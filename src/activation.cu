#include "activation.cuh"

// --- Kernels ---

// ReLU: Apply ReLU activation
__global__ void relu_kernel(double* x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = (x[idx] > 0.0) ? x[idx] : 0.0;
    }
}

// Exponential: Calculate exp(x_i - max_val)
__global__ void exp_kernel(double* x, int size, double max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        x[idx] = exp(x[idx] - max_val);
    }
}

// Normalize: Normalize array by sum or assign uniform distribution if sum is zero.
__global__ void divide_by_sum_kernel(double* x, int size, double sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size && sum != 0.0) {
        x[idx] /= sum;
    } else if (idx < size && sum == 0.0) {
        x[idx] = 1.0 / size; // Uniform distribution if sum is zero
    }
}

// --- Wrapper Functions ---

// ReLU: Apply ReLU activation function on GPU
void relu_gpu(double* d_x, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, size);
    CHECK_CUDA(cudaGetLastError()); // Kernel launch errors
    // CHECK_CUDA(cudaDeviceSynchronize()); // Uncomment for immediate checking
}

// Naive Softmax: Calculate exp on GPU, sum on CPU, divide on GPU
void softmax_gpu(double* d_x, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    double* h_x = (double*)malloc(size * sizeof(double));
    if (!h_x) {
        fprintf(stderr, "Failed to allocate host memory for softmax\n");
        exit(EXIT_FAILURE);
    }

    CHECK_CUDA(cudaMemcpy(h_x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost));

    // Find max (CPU)
    double max_val = h_x[0];
    for(int i = 1; i < size; ++i) {
        if (h_x[i] > max_val) max_val = h_x[i];
    }

    // Calculate exp(x_i - max_val) (GPU)
    exp_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, size, max_val);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(h_x, d_x, size * sizeof(double), cudaMemcpyDeviceToHost));

    // Calculate sum (CPU)
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += h_x[i];
    }

    free(h_x);

    if (sum == 0.0) {
        fprintf(stderr, "Warning: Softmax sum is zero. Setting uniform distribution.\n");
    }

    // Divide by sum (GPU)
    divide_by_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, size, sum);
    CHECK_CUDA(cudaGetLastError()); // Kernel launch errors
    // CHECK_CUDA(cudaDeviceSynchronize()); // Uncomment for immediate checking
}