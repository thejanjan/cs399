#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int THREADS = 32;

__global__ void setup_axpy(float *x, float *y, int n) {
	// calculate the x and y index the thread is working on
	int xi = threadIdx.x + blockIdx.x*blockDim.x;
	int yi = threadIdx.y + blockIdx.y*blockDim.y;
	
	if (xi < n && yi < n) {
		// get index
		int i = xi + (yi * n);
		
		// fill with initial values
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
}

__global__ void axpy(float *x, float *y, int n, float a) {
	// calculate the x and y index the thread is working on
	int xi = threadIdx.x + blockIdx.x*blockDim.x;
	int yi = threadIdx.y + blockIdx.y*blockDim.y;

	if (xi < n && yi < n) {
		// get index
		int i = xi + (yi * n);

		// do axpy
		y[i] = a*x[i]+y[i];
	}
}

void verify_results(float *y, int n, float a) {
	bool success = true;
	float output_val = a * 1.0f + 2.0f;  // a*x+y
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (y[j + (i * n)] != output_val) {
				printf("Verify Failure - at y[%d]\n", j + (i * n));
				printf("%f != %f\n", y[j + (i * n)], output_val);
				success = false;
			}
		}
	}
	if (success) {
		printf("Output verified.\n");
	}
}

int main(int argc, char *argv[]) {
	// Handle option inputs -- really rough arg parsing
	int n = 1000;
	float a = 2.0f;

	if (argc == 3) {
		if (strcmp(argv[1], "-n") == 0) {
			n = atoi(argv[2]);
		}
	}

	///
	/// AXPY operation
	///

	// setup cuda
	dim3 threads_per_block(THREADS, THREADS);
	dim3 blocks_per_grid(ceil((float)n/(float)THREADS), ceil((float)n/(float)THREADS));

	float *x_d, *y_d;
	float *y = (float *)malloc(n * n * sizeof(float));
	cudaMalloc(&x_d, n * n * sizeof(float));
	cudaMalloc(&y_d, n * n * sizeof(float));

	// perform kernels
	setup_axpy<<<blocks_per_grid, threads_per_block>>>(x_d, y_d, n);
	axpy<<<blocks_per_grid, threads_per_block>>>(x_d, y_d, n, a);

	// collect result
	cudaMemcpy(y, y_d, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
	cudaError_t errorCode = cudaGetLastError();
	if (errorCode != cudaSuccess)
	{
		printf("Cuda error %d: %s\n", errorCode, cudaGetErrorString(errorCode));
		exit(0);
	}
	cudaFree(x_d);
	cudaFree(y_d);

	// verify results
	verify_results(y, n, a);

	// return and cleanup
	free(y);
    return 0;
}
