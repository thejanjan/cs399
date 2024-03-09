#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int THREADS = 32;

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

void verify_results(float *out, int n, float a) {
	bool success = true;
	float output_val = a * 1.0f + 2.0f;  // a*x+y
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (out[j + (i * n)] != output_val) {
				printf("Verify Failure - at y[%d]\n", j + (i * n));
                printf("%f != %f\n", out[j + (i * n)], output_val);
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
	dim3 blocks_per_grid(ceil((float)numx/(float)THREADS), ceil((float)numy/(float)THREADS));
	
    float *x_d *y_d;
    float *x = (float *)malloc(n * n * sizeof(float));
	float *y = (float *)malloc(n * n * sizeof(float));
    cudaMalloc(&x_d, n * n * sizeof(float));
	cudaMalloc(&y_d, n * n * sizeof(float));

    // perform kernel
    axpy<<<blocks_per_grid, threads_per_block>>>(x_d, y_d, n, a);

    // collect result
    cudaMemcpy(x, x_d, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(y, y_d, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
    cudaFree(x_d);
	cudaFree(y_d);

    // verify results
	verify_results(y, n, a);

    // return and cleanup
    free(x);
	free(y);
    return 0;
}
