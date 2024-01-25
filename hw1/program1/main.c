#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

// Calculates a random floating point number
// between 0 and 1.
float randf() {
    return (float)rand() / (float)RAND_MAX;
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

    // algorithm constants
    float *points_x = (float *)malloc(num_points * sizeof(float));
    float *points_y = (float *)malloc(num_points * sizeof(float));
    int bounded_points = 0;

    // generate random points
    srand(time(NULL));

    // NOTE: it would be much more efficient to not have to allocate memory
    //  for this and just do calculations with bare randf() calls,
    //  but the rubric requests specifically "generate N random 2-dimensional points",
    //  so i'm interpreting it as such
    #pragma acc enter data create(points_x[0:num_points], points_y[0:num_points])

    #pragma acc parallel loop
    for (int i = 0; i < num_points; i++) {
        points_x[i] = randf();
        points_y[i] = randf();
    }

    // calculate how many points are valid
    #pragma acc parallel loop reduction(+:bounded_points)
    for (int i = 0; i < num_points; i++) {
        if (((points_x[i] * points_x[i]) + (points_y[i] * points_y[i])) <= 1.0) {
            bounded_points++;
        }
    }

    #pragma acc exit data delete(points_x, points_y)

    // calculate pi, display results
    float pi_approx = 4.0f * ((float)bounded_points / (float)num_points);
    printf("Number of points: %d\n", num_points);
    printf("Points within quarter circle: %d\n", bounded_points);
    printf("Pi approximate: %f\n", pi_approx);


    // cleanup and return
    free(points_x);
    free(points_y);
    return 0;
}
