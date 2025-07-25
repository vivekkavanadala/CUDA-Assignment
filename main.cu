#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void reduce_sum(float* input, float* output) {
    // Allocate shared memory for the block
    extern __shared__ float sdata[];
    
    // Each thread loads one element from global to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = input[i];
    __syncthreads();
    
    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    float *h_input, *h_output; // host variables
    float *d_input, *d_output; // device variables

    size_t size = N * sizeof(float);
    h_input = (float*) malloc(size);
    h_output = (float*) malloc(sizeof(float));

    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    reduce_sum<<<1, N, N * sizeof(float)>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sum = %f\n", h_output[0]);

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
