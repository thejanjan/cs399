#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int THREADS = 32;

///
/// benchmark functions
///

float _bm_startTime;

void start_benchmark() {
	_bm_startTime = (float)clock()/CLOCKS_PER_SEC;
}

void end_benchmark(char *name) {
	float _bm_endTime = (float)clock()/CLOCKS_PER_SEC;
	float timeElapsed = _bm_endTime - _bm_startTime;
	printf("-=- %s -=-\n", name);
	printf("Time elapsed: %f ms\n", timeElapsed * 1000.0f);
}

///
/// kernels
///

__global__ void matrix_init(float *a, float *a2, float *b, float *b2, float *c, int m, int n, float value, int *fails) {
	int xi = threadIdx.x + blockIdx.x*blockDim.x;
	int yi = threadIdx.y + blockIdx.y*blockDim.y;
	
	int i = xi + (yi * n);
	
	if (i < (n * n)) {
		c[i] = value;
	}
	
	if (i < (n * m)) {
		a[i] = value;
		b2[i] = value;
		b[i] = value;
		a2[i] = value;
	}
	
	if (xi == 0 && yi == 0) {
		*fails = 0;
	}
}

__global__ void matrix_test(float *C, int m, int n, int *out) {
	// calculate the x and y index the thread is working on
	int xi = threadIdx.x + blockIdx.x*blockDim.x;
	int yi = threadIdx.y + blockIdx.y*blockDim.y;

	if (xi < n && yi < n) {
		// test output
		int i = yi + (xi * n);
		if (round(C[i]) != m) {
			atomicAdd(out, 1);
		}
	}
}

__global__ void matrix_mult(float *A, float *B, float *C, int m, int n) {
	// calculate the x and y index the thread is working on
	int xi = threadIdx.x + blockIdx.x*blockDim.x;
	int yi = threadIdx.y + blockIdx.y*blockDim.y;

	if (xi < n && yi < n) {
		float result = 0.0f;
		for (int k = 0; k < m; k++) {
			int ai = k + (xi * m);
			int bi = yi + (k * n);
			result += A[ai] * B[bi];
		}

		// set output
		int i = yi + (xi * n);
		C[i] = result;
	}
}

__global__ void matrix_mult_b2(float *A, float *B2, float *C, int m, int n) {
	// calculate the x and y index the thread is working on
	int xi = threadIdx.x + blockIdx.x*blockDim.x;
	int yi = threadIdx.y + blockIdx.y*blockDim.y;

	if (xi < n && yi < n) {
		// perform sum
		float result = 0.0f;
		for (int k = 0; k < m; k++) {
			int ai = k + (xi * m);
			result += A[ai] * B2[ai];
		}

		// set output
		int i = yi + (xi * n);
		C[i] = result;
	}
}

__global__ void matrix_mult_a2(float *A2, float *B, float *C, int m, int n) {
	// calculate the x and y index the thread is working on
	int xi = threadIdx.x + blockIdx.x*blockDim.x;
	int yi = threadIdx.y + blockIdx.y*blockDim.y;

	if (xi < n && yi < n) {
		// perform sum
		float result = 0.0f;
		for (int k = 0; k < m; k++) {
			int bi = yi + (k * n);
			result += A2[bi] * B[bi];
		}

		// set output
		int i = yi + (xi * n);
		C[i] = result;
	}
}

///
/// main
///

int main(int argc, char *argv[]) {
	// Handle option inputs -- really rough arg parsing
	int n = 100;
	int m = 100;

	if (argc == 5) {
		if (strcmp(argv[1], "-n") == 0) {
			n = atoi(argv[2]);
		}
		if (strcmp(argv[3], "-m") == 0) {
			m = atoi(argv[4]);
		}
	}

	///
	/// Mandelbrot calculations
	///

	// setup cuda
	int max_dim = max(n, m);
	dim3 threads_per_block(THREADS, THREADS);
	dim3 blocks_per_grid_max(ceil((float)max_dim/(float)THREADS), ceil((float)max_dim/(float)THREADS));
	dim3 blocks_per_grid_nn(ceil((float)n/(float)THREADS), ceil((float)n/(float)THREADS));

	float *a_d, *a2_d, *b_d, *b2_d, *c_d, *c;
	c = (float *)malloc(n * n * sizeof(float));
	cudaMalloc(&a_d,  n * m * sizeof(float));
	cudaMalloc(&a2_d, n * m * sizeof(float));
	cudaMalloc(&b_d,  n * m * sizeof(float));
	cudaMalloc(&b2_d, n * m * sizeof(float));
	cudaMalloc(&c_d,  n * n * sizeof(float));
	
	int *fails_d, *fails;
	fails = (int *)malloc(sizeof(int));
	cudaMalloc(&fails_d, sizeof(int));

	// init matrices
	matrix_init<<<blocks_per_grid_max, threads_per_block>>>(a_d, a2_d, b_d, b2_d, c_d, m, n, 1.0f, fails_d);

	// do initial computation, benchmark
	start_benchmark();
	matrix_mult<<<blocks_per_grid_nn, threads_per_block>>>(a_d, b_d, c_d, m, n);
	end_benchmark("A*B");

	// check results
	matrix_test<<<blocks_per_grid_nn, threads_per_block>>>(c, m, n, fails_d);
	cudaMemcpy(fails, fails_d, sizeof(int), cudaMemcpyDeviceToHost);
	if (fails > 0) {
		printf("Verify misses: %d\n", fails);
		
		// do a more precise test
		cudaMemcpy(c, c_d, sizeof(float) * n * n, cudaMemcpyDeviceToHost);	
		for (int i = 0; i < (n * n); i++) {
			int value = round(c[i]);
			if (value != m) {
				printf("Miss at i=%d: %d (expected: %d)\n", i, value, m);
			}
		}
	}
	free(fails);
	cudaFree(fails_d);

	// transpose B, benchmark
	start_benchmark();
	matrix_mult_b2<<<blocks_per_grid_nn, threads_per_block>>>(a_d, b2_d, c_d, m, n);
	end_benchmark("A*B, transposed B");

	// transpose A, benchmark
	start_benchmark();
	matrix_mult_a2<<<blocks_per_grid_nn, threads_per_block>>>(a2_d, b_d, c_d, m, n);
	end_benchmark("A*B, transposed A");

	// return and cleanup
	free(c);
	cudaFree(a_d);
	cudaFree(a2_d);
	cudaFree(b_d);
	cudaFree(b2_d);
	cudaFree(c_d);
	return 0;
}
