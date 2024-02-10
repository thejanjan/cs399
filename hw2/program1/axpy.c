#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
}

///
/// Main
///

int main(int argc, char *argv[]) {
    // Handle option inputs -- really rough arg parsing
    int n = 1000;
    float a = 2.0f;

    if (argc == 3) {
        if (strcmp(argv[1], "-n") == 0) {
            n = atoi(argv[2]);
        }
    }

    // Initialize float arrays on CPU
    float **x = (float **)malloc(n * sizeof(float *));
    float **y = (float **)malloc(n * sizeof(float *));
    for (int i = 0; i < n; i++) {
        x[i] = (float *)malloc(n * sizeof(float));
        y[i] = (float *)malloc(n * sizeof(float));
    }

    // Data region for 2d arrays on GPU
    start_benchmark();
    #pragma acc data create(x[:n][:n], y[:n][:n])
    {
        // Set float values on GPU
        #pragma acc parallel loop gang
        for (int i = 0; i < n; i++) {
            #pragma acc loop vector
            for (int j = 0; j < n; j++) {
                x[i][j] = 1.0f;
                y[i][j] = 2.0f;
            }
        }

        // Compute sum over array
        #pragma acc parallel loop gang
        for (int i = 0; i < n; i++) {
            #pragma acc loop vector
            for (int j = 0; j < n; j++) {
                y[i][j] = a*x[i][j]+y[i][j];
            }
        }

        // Copy result back to CPU
        #pragma acc update self(x[:n][:n], y[:n][:n])
    }
    end_benchmark("GPU process time", n);

    // cleanup and return
    for (int i = 0; i < n; i++) {
        free(x[i]);
        free(y[i]);
    }
    free(x);
    free(y);
    return 0;
}
