#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int THREADS = 32;

__global__ void heat_init(float *a, int n) {
    // calculate the x and y index the thread is working on
	int xi = threadIdx.x + blockIdx.x*blockDim.x;
	int yi = threadIdx.y + blockIdx.y*blockDim.y;
	
	if (xi < n && yi < n) {
		// get index
		int i = xi + (yi * n);
		
		// initialize array
		a[i] = (yi == 0) ? 100.0f : 0.0f;
	}
}

__global__ void heat_iteration(float *a, float *b, int n, float *err) {
    // calculate the x and y index the thread is working on
	int xi = threadIdx.x + blockIdx.x*blockDim.x;
	int yi = threadIdx.y + blockIdx.y*blockDim.y;
	
	if (xi > 0 && yi > 0 && xi < (n - 1) && yi < (n - 1)) {
		// get index
		int i = xi + (yi * n);
		int left = xi - 1 + (yi * n);
		int right = xi + 1 + (yi * n);
		int up = xi + ((yi - 1) * n);
		int down = xi + ((yi + 1) * n);
		
		// do heat iteration
		b[i] = (a[right] + a[left] + a[up] + a[down]) / 4;
		
		// measure error
		err[i] = abs(b[i] - a[i]);
	}
}

void checkForCudaError() {
	cudaError_t errorCode = cudaGetLastError();
	if (errorCode != cudaSuccess)
	{
		printf("Cuda error %d: %s\n", errorCode, cudaGetErrorString(errorCode));
		exit(0);
	}
}

int main(int argc, char *argv[]) {
    // Handle option inputs -- really rough arg parsing
    int n = 256;
    float tol = 0.01f;
    int max_iter = 3000;

    if (argc == 7) {
        if (strcmp(argv[1], "-n") == 0) {
            n = atoi(argv[2]);
        }
        if (strcmp(argv[3], "-tol") == 0) {
            tol = atof(argv[4]);
        }
        if (strcmp(argv[5], "-max_iter") == 0) {
            max_iter = atoi(argv[6]);
        }
    }

    ///
    /// AXPY operation
    ///

    // setup cuda
	printf("1\n");
	dim3 threads_per_block(THREADS, THREADS);
	dim3 blocks_per_grid(ceil((float)n/(float)THREADS), ceil((float)n/(float)THREADS));
	
	// using A and B instead of A and Anew,
	// since I'm swapping their pointers for efficiency
    float *a_d, *b_d, *err_d;
    float *out = (float *)malloc(n * n * sizeof(float));
	float *err = (float *)malloc(n * n * sizeof(float));
    cudaMalloc(&a_d, n * n * sizeof(float));
	cudaMalloc(&b_d, n * n * sizeof(float));
	cudaMalloc(&err_d, n * n * sizeof(float));
	checkForCudaError();
	printf("2\n");

	// setup the heat arrays
	heat_init<<<blocks_per_grid, threads_per_block>>>(a_d, n);
	heat_init<<<blocks_per_grid, threads_per_block>>>(b_d, n);
	printf("3\n");
	
	// now begin the heat loop, loop over each iteration
	for (int iter = 1; iter <= max_iter; iter++) {
		// perform iteration step
		heat_iteration<<<blocks_per_grid, threads_per_block>>>(a_d, b_d, n, err_d);
		checkForCudaError();

		// attempt printing to new csv
		if ((iter % 1000) == 0) {
			printf("4\n");
			// get data from gpu
			cudaMemcpy(out, b_d, sizeof(float) * n * n, cudaMemcpyDeviceToHost);

			// print it out as csv
			char *fname = (char *)malloc(20 * sizeof(char));
			if (fname == NULL) {
				printf("Error when allocating filename");
				continue;
			}
			snprintf(fname, 20, "heat_%d.csv", iter);
			FILE *fp = fopen(fname, "w");
			if (fp == NULL) {
				free(fname);
				printf("Error when creating csv");
				continue;
			}
			for (int y = 0; y < n; y++){
				for (int x = 0; x < n; x++){
					fprintf(fp, "%f%s",
						out[x + (iter * n)],       // write number
						((x + 1) == n) ? "\n" : ","  // add comma, unless end of line
					);
				}
			}
			fclose(fp);
			free(fname);
			printf("5\n");
		}

		// check that heat is within tolerance
		cudaMemcpy(err, err_d, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
		checkForCudaError();
		float measured_err = 0.0;
		for (int i = 0; i < (n * n); i++) {
			measured_err = max(err[i], measured_err);
		}
		if (measured_err <= tol) {
			printf("Tolerance reached at iteration %d, (%f <= %f)\n", iter, measured_err, tol);
			break;
		}
		
		// swap pointers
		float *temp = a_d;
		a_d = b_d;
		b_d = temp;
	}
	printf("6\n");
	
	// free memory
    cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(err_d);
	checkForCudaError();
    free(out);
	free(err);
    return 0;
}
