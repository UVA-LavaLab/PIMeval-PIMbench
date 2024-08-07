#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cublas.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime.h>

// CUDA and CUBLAS functions
#include "helper_cuda.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/pair.h>
#include <thrust/iterator/constant_iterator.h>
#include <iostream>

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

struct manhatten_distance_functor {
    const float* query;
    const float* ref;
    int dim;

    manhatten_distance_functor(const float* _query, const float* _ref, int _dim)
        : query(_query), ref(_ref), dim(_dim) {}

    __device__ float operator()(int idx) const {
        float dist = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float diff = ref[idx * dim + i] - query[i];
            dist += fabsf(diff);
        }
        return dist;
    }
};

// Functor to extract the label from the last dimension of the reference data
struct label_extraction_functor {
    const float* ref;
    int dim;

    label_extraction_functor(const float* _ref, int _dim)
        : ref(_ref), dim(_dim) {}

    __device__ int operator()(int idx) const {
        return static_cast<int>(ref[idx * dim + (dim - 1)]);
    }
};

bool knn(const float * ref,
                     int           ref_nb,
                     const float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     int *         query_labels) {


    // Copy data to device
    thrust::device_vector<float> d_ref(ref, ref + ref_nb * dim);
    thrust::device_vector<float> d_query(query, query + query_nb * dim);
    // Vectors to store k nearest neighbors
    thrust::device_vector<int> indices;
    thrust::device_vector<float> distances;
    thrust::device_vector<int> classifications;
    // Extract top k distances and corresponding indices
    indices.resize(query_nb * k);
    distances.resize(query_nb * k);
    classifications.resize(query_nb);
    // Allocate memory for distances and indices
    thrust::device_vector<float> dist_matrix(query_nb * ref_nb);
    thrust::device_vector<int> index_matrix(query_nb * ref_nb);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    // Compute distances
    for (int i = 0; i < query_nb; ++i) {
        const float* query_ptr = thrust::raw_pointer_cast(&d_query[i * dim]);
        const float* ref_ptr = thrust::raw_pointer_cast(d_ref.data());
        manhatten_distance_functor dist_functor(query_ptr, ref_ptr, dim);

        thrust::transform(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(ref_nb),
            dist_matrix.begin() + i * ref_nb,
            dist_functor
        );
        // Initialize the indices for each row
        thrust::sequence( index_matrix.begin() + i*ref_nb, index_matrix.begin() + (i + 1) * ref_nb);
    }
    cudaDeviceSynchronize();


    // Sort the distances and indices
    for (int i = 0; i < query_nb; ++i) {
        thrust::sort_by_key(dist_matrix.begin() + i * ref_nb, dist_matrix.begin() + (i + 1) * ref_nb, index_matrix.begin() + i * ref_nb);
    }

    // Gather the k-nearest neighbor labels
    thrust::device_vector<int> knn_labels(query_nb * k);
    label_extraction_functor label_functor(thrust::raw_pointer_cast(d_ref.data()), dim);

    for (int i = 0; i < query_nb; ++i) {
        thrust::copy(dist_matrix.begin() + i * ref_nb, dist_matrix.begin() + i * ref_nb + k, distances.begin() + i * k);
        thrust::copy(index_matrix.begin() + i * ref_nb, index_matrix.begin() + i * ref_nb + k, indices.begin() + i * k);
    }

    for (int i = 0; i < query_nb; ++i) {

        if (i * ref_nb + k > index_matrix.size() || i * k + k > knn_labels.size()) {
            std::cerr << "Error: Index out of bounds before transform!" << std::endl;
            return false;
        }
        thrust::transform(
            index_matrix.begin() + i * ref_nb,
            index_matrix.begin() + i * ref_nb + k,
            knn_labels.begin() + i * k,
            label_functor
        );
    }

    // Count the occurrence of each label and classify based on the majority label
    thrust::device_vector<int> label_counts(k);

    for (int i = 0; i < query_nb; ++i) {
        thrust::device_vector<int> unique_labels(k);
        thrust::device_vector<int> label_counts(k);

        // Sort the k-nearest labels to group identical labels together
        thrust::sort(knn_labels.begin() + i * k, knn_labels.begin() + i * k + k);

        // Reduce by key: Count the occurrence of each label
        auto end_pair = thrust::reduce_by_key(knn_labels.begin() + i * k,
                                            knn_labels.begin() + i * k + k,
                                            thrust::constant_iterator<int>(1),
                                            unique_labels.begin(),
                                            label_counts.begin());

        int num_unique_labels = end_pair.first - unique_labels.begin();

        // Find the label with the maximum count
        int max_label_idx = thrust::max_element(label_counts.begin(), label_counts.begin() + num_unique_labels) - label_counts.begin();

        // Assign the classification for this query
        classifications[i] = unique_labels[max_label_idx];
    }
    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));
    float msecTotal = 0.0f;

    // Record the stop event
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    printf("Execution time of k nearest neighbor = %f ms\n", msecTotal);
    // Copy data to host
    thrust::copy(classifications.begin(), classifications.end(), query_labels);

    return true;
}


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

    printf("TESTS\n");
    // Allocate memory for computed k-NN neighbors
    int   * test_knn_result = (int*)   malloc(query_nb * sizeof(int));

    // Allocation check
    if (!test_knn_result) {
        printf("ALLOCATION ERROR\n");
        free(test_knn_result);
        return false;
    }

    // See if knn returns any errors
    if (!knn(ref, ref_nb, query, query_nb, dim, k, test_knn_result)) {
        free(test_knn_result);
        return false;
    }


    // Deallocate memory 
    free(ref);
    free(query);
    free(knn_dist);
    free(knn_index);
    free(test_knn_result);
}