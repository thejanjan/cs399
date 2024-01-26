#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

    // Algorithm constants
    int max_iterations = 1000;
    int *iter_out = (int *)malloc(numx * numy * sizeof(int));

    // Iterate over each grid point.
    #pragma acc parallel loop copyout(iter_out[0:numx*numy])
    for (int i = 0; i < (numx * numy); i++) {
        // Calculate x and y index.
        int xi = i % numx;
        int yi = i / numy;

        // Calculate x and y coordinates.
        float x0 = (((float)xi / (float)numx) * 3.0f) - 2.0f;
        float y0 = (((float)yi / (float)numy) * 4.0f) - 2.0f;

        // Run algorithm and its constants.
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
        iter_out[i] = iteration;
    }

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
