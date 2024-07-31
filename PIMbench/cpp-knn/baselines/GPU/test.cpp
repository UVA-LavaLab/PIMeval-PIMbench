#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

#include "knncuda.h"
//------------------
#include <VaryTypeAndOperator.h>
#include <PrintOutput.h>

MultiLineExternalVariablesMacros

typedef struct Params
{
    int numTestPoints;
    int numDataPoints;
    int dimension;
    int k;
    char *inputTestFile;
    char *inputDataFile;
} Params;

void usage()
{
    fprintf(stderr,
            "\nUsage:  ./program [options]"
            "\n"
            "\n    -n    number of data points (default=65536 points)"
            "\n    -m    number of test points (default=100 points)"
            "\n    -d    dimension (default=2)"
            "\n    -k    value of K (default=20)"
            "\n    -i    input file containing training datapoints (default=generates datapoints with random numbers)"
            "\n    -j    input file containing testing datapoints (default=generates datapoints with random numbers)"
            "\n");
}

struct Params input_params(int argc, char **argv)
{
    struct Params p;
    p.numDataPoints = 65536;
    p.numTestPoints = 100;
    p.dimension = 2;
    p.k = 20;
    p.inputTestFile = nullptr;
    p.inputDataFile = nullptr;

    int opt;
    while ((opt = getopt(argc, argv, "h:k:n:m:d:i:j:")) >= 0)
    {
        switch (opt)
        {
        case 'h':
            usage();
            exit(0);
            break;
        case 'n':
            p.numDataPoints = atoll(optarg);
            break;
        case 'm':
            p.numTestPoints = atoll(optarg);
            break;
        case 'd':
            p.dimension = atoll(optarg);
            break;
        case 'k':
            p.k = atoi(optarg);
            break;
        case 'i':
            p.inputDataFile = optarg;
            break;
        case 'j':
            p.inputTestFile = optarg;
            break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
        }
    }

    return p;
}


/**
 * Initializes randomly the reference and query points.
 *
 * @param ref        refence points
 * @param ref_nb     number of reference points
 * @param query      query points
 * @param query_nb   number of query points
 * @param dim        dimension of points
 */
void initialize_data(float * ref,
                     int     ref_nb,
                     float * query,
                     int     query_nb,
                     int     dim) {

    // Initialize random number generator
    srand(time(NULL));

    // Generate random reference points
    for (int i=0; i<ref_nb*dim; ++i) {
        ref[i] = 10. * (float)(rand() / (double)RAND_MAX);
    }

    // Generate random query points
    for (int i=0; i<query_nb*dim; ++i) {
        query[i] = 10. * (float)(rand() / (double)RAND_MAX);
    }
}

/**
 * Test an input k-NN function implementation by verifying that its output
 * results (distances and corresponding indexes) are similar to the expected
 * results (ground truth).
 *
 * Since the k-NN computation might end-up in slightly different results
 * compared to the expected one depending on the considered implementation,
 * the verification consists in making sure that the accuracy is high enough.
 *
 *
 * @param ref            reference points
 * @param ref_nb         number of reference points
 * @param query          query points
 * @param query_nb       number of query points
 * @param dim            dimension of reference and query points
 * @param k              number of neighbors to consider
 * @param knn            function to test
 * @param name           name of the function to test (for display purpose)
 * return false in case of problem, true otherwise
 */
bool test(const float * ref,
          int           ref_nb,
          const float * query,
          int           query_nb,
          int           dim,
          int           k,
          bool (*knn)(const float *, int, const float *, int, int, int, int *, double &),
          double &elapsed_time) {

    // Display k-NN function name
    printf("- global memory: ");

    // Allocate memory for computed k-NN neighbors
    int   * test_knn_result = (int*)   malloc(query_nb * sizeof(int));

    // Allocation check
    if (!test_knn_result) {
        printf("ALLOCATION ERROR\n");
        free(test_knn_result);
        return false;
    }
    // Start timer
    struct timeval tic;
    gettimeofday(&tic, NULL);

    if (!knn(ref, ref_nb, query, query_nb, dim, k, test_knn_result, elapsed_time)) {


        free(test_knn_result);
        return false;
    }


    // Stop timer
    struct timeval toc;
    gettimeofday(&toc, NULL);

    // Elapsed time in ms
    elapsed_time = toc.tv_sec - tic.tv_sec;
    elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;

    timeInMsec=(elapsed_time)*1000.0f;

    sizeInGBytes= (ref_nb   * dim * sizeof(float)+ query_nb * dim * sizeof(float)  )* 1.0e-9;
    outPutSizeInGBytes=(query_nb * k   * sizeof(float)+query_nb * k   * sizeof(int))*1.0e-9;
    if(timeInMsec!=0){
    	  gigaProcessedInSec=( sizeInGBytes) / (timeInMsec / 1000.0f);
   }
    printOutput();

    // Free memory
    free(test_knn_result);

    return true;
}


/**
 * 1. Create the synthetic data (reference and query points).
 * 2. Compute the ground truth.
 * 3. Test the different implementation of the k-NN algorithm.
 */

int main(int argc, char **argv)
{
    struct Params p = input_params(argc, argv);
    // Parameters
    int ref_nb   = p.numDataPoints;
    int query_nb = p.numTestPoints;
    int dim      = p.dimension;
    int k        = p.k;

    // Display

    printf("PARAMETERS\n");
         printf("- Number reference points : %d\n",   ref_nb);
         printf("- Number query points     : %d\n",   query_nb);
         printf("- Dimension of points     : %d\n",   dim);
         printf("- Number of neighbors     : %d\n\n", k);
    // Sanity check
    if (ref_nb<k) {
        printf("Error: k value is larger that the number of reference points\n");
        return EXIT_FAILURE;
    }
    double elapsed_time;
    // Allocate input points and output k-NN distances / indexes
    float * ref        = (float*) malloc(ref_nb   * dim * sizeof(float));
    if (!ref) {
        printf("Error allocating ref: %s\n", strerror(errno));
        return EXIT_FAILURE;
    }
    float * query      = (float*) malloc(query_nb * dim * sizeof(float));
    if (!query) {
        printf("Error allocating query: %s\n", strerror(errno));
        free(ref);
        return EXIT_FAILURE;
    }
    float * knn_dist   = (float*) malloc(query_nb * k   * sizeof(float));
    if (!knn_dist) {
        printf("Error allocating knn_dist: %s\n", strerror(errno));
        free(ref);
        free(query);
        return EXIT_FAILURE;
    }
    int   * knn_index  = (int*)   malloc(query_nb * k   * sizeof(int));
    if (!knn_index) {
        printf("Error allocating knn_index: %s\n", strerror(errno));
        free(ref);
        free(query);
        free(knn_dist);
        return EXIT_FAILURE;
    }


    // Initialize reference and query points with random values
    initialize_data(ref, ref_nb, query, query_nb, dim);


    // Test k-NN function
    printf("TESTS\n");

    test(ref, ref_nb, query, query_nb, dim, k, &knn_cuda_global, elapsed_time);

    // Deallocate memory 
    free(ref);
    free(query);
    free(knn_dist);
    free(knn_index);
#ifndef METRIC_RUN_MAIN
    return EXIT_SUCCESS;
#else
    return true;
#endif

}
