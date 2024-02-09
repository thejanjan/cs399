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

void end_benchmark(char *name) {
    float _bm_endTime = (float)clock()/CLOCKS_PER_SEC;
    float timeElapsed = _bm_endTime - _bm_startTime;
    printf("-=- %s -=-\n", name);
    printf("Time elapsed: %f ms\n", timeElapsed * 1000.0f);
}

///
/// Main
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

    // Initialize matrices
    float **A = (float **)malloc(n * sizeof(float *));
    float **A2 = (float **)malloc(m * sizeof(float *));
    float **B = (float **)malloc(m * sizeof(float *));
    float **B2 = (float **)malloc(n * sizeof(float *));
    float **C = (float **)malloc(n * sizeof(float *));
    for (int i = 0; i < n; i++) {
        A[i] = (float *)malloc(m * sizeof(float));
        B2[i] = (float *)malloc(m * sizeof(float));
        C[i] = (float *)malloc(n * sizeof(float));
        for (int j = 0; j < m; j++) {
            A[i][j] = 1.0f;
            B2[i][j] = 1.0f;
        }
    }
    for (int i = 0; i < m; i++) {
        B[i] = (float *)malloc(n * sizeof(float));
        A2[i] = (float *)malloc(n * sizeof(float));
        for (int j = 0; j < n; j++) {
            B[i][j] = 1.0f;
            A2[i][j] = 1.0f;
        }
    }

    // part b: compute the product
    start_benchmark();
    #pragma acc parallel loop gang copyin(A[0:n][0:m], B[0:m][0:n]) copyout(C[0:n][0:n])
    for (int i = 0; i < n; i++) {
        #pragma acc loop worker
        for (int j = 0; j < n; j++) {
            float result = 0.0f;
            #pragma acc loop vector reduction(+:result)
            for (int k = 0; k < m; k++) {
                result += A[i][k]*B[k][j];
            }
            C[i][j] = result;
        }
    }
    end_benchmark("A*B");

    // part c: confirm result
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (C[i][j] != m) {
                printf("invalid result at [%d][%d]: %f\n", i, j, C[i][j]);
            }
        }
    }

    // part d: use transposed B
    start_benchmark();
    #pragma acc parallel loop gang copyin(A[0:n][0:m], B2[0:n][0:m]) copyout(C[0:n][0:n])
    for (int i = 0; i < n; i++) {
        #pragma acc loop worker
        for (int j = 0; j < n; j++) {
            float result = 0.0f;
            #pragma acc loop vector reduction(+:result)
            for (int k = 0; k < m; k++) {
                result += A[i][k]*B2[j][k];
            }
            C[i][j] = result;
        }
    }
    end_benchmark("A*B, transposed B");

    // part e: use transposed A
    start_benchmark();
    #pragma acc parallel loop gang copyin(A2[0:m][0:n], B[0:m][0:n]) copyout(C[0:n][0:n])
    for (int i = 0; i < n; i++) {
        #pragma acc loop worker
        for (int j = 0; j < n; j++) {
            float result = 0.0f;
            #pragma acc loop vector reduction(+:result)
            for (int k = 0; k < m; k++) {
                result += A2[k][i]*B[k][j];
            }
            C[i][j] = result;
        }
    }
    end_benchmark("A*B, transposed A");

    // Cleanups
    for (int i = 0; i < n; i++) {
        free(A[i]);
        free(B2[i]);
        free(C[i]);
    }
    for (int i = 0; i < m; i++) {
        free(B[i]);
        free(A2[i]);
    }
    free(A);
    free(A2);
    free(B);
    free(B2);
    free(C);
    return 0;
}
