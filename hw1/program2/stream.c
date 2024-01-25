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
    printf("Estimated bandwidth: %f elements/sec\n\n", (float)array_size / timeElapsed);
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

    ///
    /// Stream benchmarks
    ///

    printf("Initializing arrays... \n");
    float *a = (float *)malloc(size * sizeof(float));
    float *b = (float *)malloc(size * sizeof(float));
    float *c = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
        c[i] = 0.0f;
    }

    float scalar = 2.0f;
    printf("Initializing arrays... DONE\n");

    /// Copy benchmark - copy A to C ///
    start_benchmark();
    #pragma acc parallel loop copyin(a[0:size]) copyout(c[0:size])
    for (int i = 0; i < size; i++) {
        c[i] = a[i];
    }
    end_benchmark("Copy Benchmark", size);

    /// Scale benchmark - scale C by X, into B ///
    start_benchmark();
    #pragma acc parallel loop copyin(c[0:size]) copyout(b[0:size])
    for (int i = 0; i < size; i++) {
        b[i] = scalar*c[i];
    }
    end_benchmark("Scale Benchmark", size);

    /// Add benchmark - add A and B, into C ///
    start_benchmark();
    #pragma acc parallel loop copyin(a[0:size], b[0:size]) copyout(c[0:size])
    for (int i = 0; i < size; i++) {
        c[i] = a[i]+b[i];
    }
    end_benchmark("Add Benchmark", size);

    /// Triad benchmark - add B and C' (C scaled by X), into A ///
    start_benchmark();
    #pragma acc parallel loop copyin(b[0:size], c[0:size]) copyout(a[0:size])
    for (int i = 0; i < size; i++) {
        a[i] = b[i]+scalar*c[i];
    }
    end_benchmark("Triad Benchmark", size);

    // return and cleanup
    free(a);
    free(b);
    free(c);
    return 0;
}
