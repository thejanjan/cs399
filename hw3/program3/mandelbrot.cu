#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const THREADS = 256;
const MAX_ITER = 1000;

__global__ void mandelbrot(int *iter_out, int numx, int numy) {
    // get thread idx
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= (numx * numy)) return;

    // calculate the x and y index the thread is working on
    int xi = idx % numx;
    int yi = idx / numy;

    // calculate x and y coordinates
    float x0 = (((float)xi / (float)numx) * 3.0f) - 2.0f;
    float y0 = (((float)yi / (float)numy) * 4.0f) - 2.0f;

    // perform algorithm and its constants
    float x = 0.0f;
    float y = 0.0f;
    float xtemp;
    int iteration = 0;
    while (((x*x + y*y) <= 4.0f) && (iteration < max_iterations)) {
        xtemp = (x*x) - (y*y) + x0;
        y = (2*x*y) + y0;
        x = xtemp;
        iteration++;
    }
    iter_out[idx] = iteration;
}

int main(int argc, char *argv[]) {
    // Handle option inputs -- really rough arg parsing
    int numx = 100;
    int numy = 100;

    if (argc == 5) {
        if (strcmp(argv[1], "-numx") == 0) {
            numx = atoi(argv[2]);
        }
        if (strcmp(argv[3], "-numy") == 0) {
            numy = atoi(argv[4]);
        }
    }

    ///
    /// Mandelbrot calculations
    ///

    // setup cuda
    int BLOCKS = (size + THREADS - 1);
    int *iter_out, *iter_out_d;
    iter_out = (int *)malloc(numx * numy * sizeof(int));
    cudaMalloc(iter_out_d, numx * numy * sizeof(int));

    // perform kernel
    mandelbrot<<<BLOCKS, THREADS>>>(iter_out_d, numx, numy);

    // collect result
    cudaMemcpy(iter_out, iter_out_d, sizeof(int) * numx * numy, cudaMemcpyDeviceToHost);
    cudaFree(iter_out_d);

    // write to csv
    FILE *fp = fopen("mandelbrot.csv", "w");
    if (fp == NULL) {
        free(iter_out);
        printf("Error when creating csv");
        return 1;
    }
    for (int y = 0; y < numy; y++){
        for (int x = 0; x < numx; x++){
            fprintf(fp, "%d%s",
                iter_out[x + (y * numx)],       // write number
                ((x + 1) == numx) ? "\n" : ","  // add comma, unless end of line
            );
        }
    }
    fclose(fp);

    // return and cleanup
    free(iter_out);
    return 0;
}