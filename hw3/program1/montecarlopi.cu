#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

const int THREADS = 16;

__global__ void monte_carlo(int *point_counts) {
    // shared memory for the threads in this block
    __shared__ int successes[THREADS];

    // some consts for this function
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // setup cuRAND
    curandState state;
    curand_init((unsigned long long)clock(), idx, 0, &state);

    // calculate success for this thread
    float x = curand_uniform(&state);
    float y = curand_uniform(&state);
    successes[threadIdx.x] = 0;
    if (((x * x) + (y * y)) < 1.0)
        successes[threadIdx.x] = 1;

    // sync threads
    __syncthreads();

    // let one thread populate the point count for the block
    if (threadIdx.x == 0) {
        point_counts[blockIdx.x] = 0;
        for (int i = 0; i < THREADS; i++)
            point_counts[blockIdx.x] += successes[i];
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
    point_counts = (int *)malloc(num_points * sizeof(int));
    cudaMalloc(&d_point_counts, num_points * sizeof(int));
	printf("Memory allocated\n");

    // perform kernel
    monte_carlo<<<BLOCKS, THREADS>>>(d_point_counts);
	printf("Kernel performed\n");

    // collect result
    cudaMemcpy(point_counts, d_point_counts, sizeof(int) * num_points, cudaMemcpyDeviceToHost);
    cudaFree(d_point_counts);
	printf("Memory freed\n");

    // count the number of successes of these values
    int bounded_points = 0;
    for (int i = 0; i < BLOCKS; i++)
        bounded_points += point_counts[i];

    // calculate pi, display results
    float pi_approx = 4.0f * ((float)bounded_points / (float)num_points);
    printf("Number of points: %d\n", num_points);
    printf("Points within quarter circle: %d\n", bounded_points);
    printf("Pi approximate: %f\n", pi_approx);

    // cleanup and return
    free(point_counts);
    return 0;
}
