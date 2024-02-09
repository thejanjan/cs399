#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// fmaxf isn't present on my linux C for some reason,
// so adding this here as a workaround
#ifndef fmaxf
float fmaxf(float x, float y) {
    if (x < y) return y;
    return x;
}
#endif


///
/// Main
///

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

    // Initialize float arrays on CPU
    float **A = (float **)malloc(n * sizeof(float *));
    float **Anew = (float **)malloc(n * sizeof(float *));
    for (int i = 0; i < n; i++) {
        A[i] = (float *)malloc(n * sizeof(float));
        Anew[i] = (float *)malloc(n * sizeof(float));
    }
    for (int i = 0; i < n; i++) {
        A[0][i] = 100.0f;
        Anew[0][i] = 100.0f;
    }

    // Loop over each iteration
    float err;
    #pragma acc data copyin(A[0:n][0:n], Anew[0:n][0:n])
	{
		for (int iter = 1; iter <= max_iter; iter++) {
			err = 0.0f;

			// Initial heat displacement loop
			#pragma acc parallel loop gang reduction(max:err)
			for (int i = 1; i < (n-1); i++) {
				#pragma acc loop vector reduction(max:err)
				for (int j = 1; j < (n-1); j++) {
					Anew[i][j] = (A[i+1][j] + A[i-1][j] + A[i][j-1] + A[i][j+1])/4;
					err = fmaxf(err, fabs(Anew[i][j] - A[i][j]));
				}
			}

			// Copy Anew into A
			#pragma acc parallel loop gang
			for (int i = 1; i < n-1; i++) {
				#pragma acc loop vector
				for (int j = 1; j < n-1; j++) {
					A[i][j] = Anew[i][j];
				}
			}

			// Are we printing a csv file?
			if ((iter % 1000) == 0) {
				// Yes, so do it.
				// Sync memory to CPU.
				#pragma acc update self(Anew[0:n][0:n])

				// Now print it out as csv.
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
							Anew[y][x],       // write number
							((x + 1) == n) ? "\n" : ","  // add comma, unless end of line
						);
					}
				}
				fclose(fp);
				free(fname);
			}

			// If the heat has not changed within tolerance, break loop
			if (err <= tol) {
				printf("Tolerance reached at iteration %d\n", iter);
				break;
			}
		}
	}

    // Cleanups
    for (int i = 0; i < n; i++) {
        free(A[i]);
        free(Anew[i]);
    }
    free(A);
    free(Anew);
    return 0;
}
