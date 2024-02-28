#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

const int THREADS = 16;

__global__ void monte_carlo(int *point_counts) {
    // some consts for this function
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // setup cuRAND
    curandState state;
    curand_init((unsigned long long)clock(), idx, 0, &state);

    // calculate success for this thread
    float x = curand_uniform(&state);
    float y = curand_uniform(&state);
    point_counts[idx] = 0;
    if (((x * x) + (y * y)) < 1.0)
        point_counts[idx] = 1;
}

__global__ void reduce(int *gdata, int *out, int N) {
	// grid-strided reduction code from lecture 5
	__shared__ float sdata[THREADS];
	int tid = threadIdx.x;
	sdata[tid] = 0;
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	while (idx < N) {
		sdata[tid] += gdata[idx];
		idx += gridDim.x*blockDim.x;
	}
	for (unsigned int s=blockDim.x/2;s>0;s>>=1) {
		__syncthreads();
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
	}
	if (tid == 0) {
		atomicAdd(out, sdata[0]);
	}
}

int main(int argc, char *argv[]) {
    // Handle option inputs -- really rough arg parsing
    int num_points = 1000;
    if (argc == 3) {
        if (strcmp(argv[1], "-numpoints") == 0) {
            num_points = atoi(argv[2]);
        }
    }

    ///
    /// Monte Carlo algorithm
    ///
	
	printf("Running with %d points\n", num_points);

    // allocate host and device memory
    int BLOCKS = (num_points + THREADS - 1);

    int *point_counts, *d_point_counts;
	int *out, *d_out;
    point_counts = (int *)malloc(num_points * sizeof(int));
	out = (int *)malloc(sizeof(int));
    cudaMalloc(&d_point_counts, num_points * sizeof(int));
	cudaMalloc(&d_out, sizeof(int));
	printf("Memory allocated\n");

    // perform kernel
    monte_carlo<<<BLOCKS, THREADS>>>(d_point_counts);
	reduce<<<BLOCKS, THREADS>>>(d_point_counts, d_out, num_points);
	printf("Kernel performed\n");

    // collect result
    cudaMemcpy(out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_point_counts);
	cudaFree(d_out);
	printf("Memory freed\n");

    // calculate pi, display results
    float pi_approx = 4.0f * ((float)(*out) / (float)num_points);
    printf("Number of points: %d\n", num_points);
    printf("Points within quarter circle: %d\n", *out);
    printf("Pi approximate: %f\n", pi_approx);

    // cleanup and return
    free(point_counts);
	free(out);
    return 0;
}
