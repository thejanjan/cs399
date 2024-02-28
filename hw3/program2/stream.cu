#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>

const int THREADS = 256;

///
/// Benchmark functions
///

float _bm_startTime;

void start_benchmark() {
    _bm_startTime = (float)clock()/CLOCKS_PER_SEC;
}

void end_benchmark(char *name, int array_size) {
    float _bm_endTime = (float)clock()/CLOCKS_PER_SEC;
    float timeElapsed = _bm_endTime - _bm_startTime;
    printf("-=- %s - size %d -=-\n", name, array_size);
    printf("Time elapsed: %f ms\n", timeElapsed * 1000.0f);
    printf("Estimated bandwidth: %f elements/sec\n\n", (float)array_size / timeElapsed);
}

///
/// Kernels
///

__global__ void initalize_float_array(int size, float *array, float value) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        array[idx] = value;
}

__global__ void initalize_double_array(int size, double *array, double value) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        array[idx] = value;
}

// Copy each element of A into C
__global__ void copy_benchmark (int size, float *a, float *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        c[idx] = a;
}

// Multiply each element of C by a scalar, then put into B
__global__ void scale_benchmark(int size, float *b, float *c, float scalar) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        b[idx] = c[idx] * scalar;
}

// Add arrays A and B and put into C
__global__ void add_benchmark  (int size, float *a, float *b, float *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        c[idx] = a[idx] + b[idx];
}

// Add array B to scaled C, then put result in A
__global__ void triad_benchmark(int size, float *a, float *b, float *c, float scalar) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
        a[idx] = b[idx]+(scalar*c[idx]);
}

///
/// Main
///

int main(int argc, char *argv[]) {
    // Handle option inputs -- really rough arg parsing
    int size = 1000;

    if (argc == 3) {
        if (strcmp(argv[1], "-size") == 0) {
            size = atoi(argv[2]);
        }
    }

    // get constants
    int BLOCKS = (size + THREADS - 1);
    float scalar = 2.0f;

    // setup device memory
    float *d_a, *d_b, *d_c;
    double *d_ad, *d_bd, *d_cd;
    cudaMalloc(&d_a, size * sizeof(float));
    cudaMalloc(&d_b, size * sizeof(float));
    cudaMalloc(&d_c, size * sizeof(float));
    cudaMalloc(&d_ad, size * sizeof(double));
    cudaMalloc(&d_bd, size * sizeof(double));
    cudaMalloc(&d_cd, size * sizeof(double));

    // benchmark the array initializations
    start_benchmark();
    initalize_float_array<<<BLOCKS, THREADS>>>(size, d_a, (float)1.0);
    initalize_float_array<<<BLOCKS, THREADS>>>(size, d_b, (float)2.0);
    initalize_float_array<<<BLOCKS, THREADS>>>(size, d_c, (float)0.0);
    end_benchmark("Initialize float arrays", size);
    
    start_benchmark();
    initalize_double_array<<<BLOCKS, THREADS>>>(size, d_ad, (double)1.0);
    initalize_double_array<<<BLOCKS, THREADS>>>(size, d_bd, (double)2.0);
    initalize_double_array<<<BLOCKS, THREADS>>>(size, d_cd, (double)0.0);
    end_benchmark("Initialize double arrays", size);

    /// Copy benchmark - copy A to C ///
    start_benchmark();
    copy_benchmark<<<BLOCKS, THREADS>>>(size, d_a, d_c);
    end_benchmark("Copy Benchmark", size);

    /// Scale benchmark - scale C by X, into B ///
    start_benchmark();
    scale_benchmark<<<BLOCKS, THREADS>>>(size, d_b, d_c, scalar);
    end_benchmark("Scale Benchmark", size);

    /// Add benchmark - add A and B, into C ///
    start_benchmark();
    add_benchmark<<<BLOCKS, THREADS>>>(size, d_a, d_b, d_c);
    end_benchmark("Add Benchmark", size);

    /// Triad benchmark - add B and C' (C scaled by X), into A ///
    start_benchmark();
    triad_benchmark<<<BLOCKS, THREADS>>>(size, d_a, d_b, d_c, scalar);
    end_benchmark("Triad Benchmark", size);

    // return and cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_ad);
    cudaFree(d_bd);
    cudaFree(d_cd);
    return 0;
}
