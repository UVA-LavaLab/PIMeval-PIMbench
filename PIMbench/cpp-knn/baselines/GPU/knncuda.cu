#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime.h>
//#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
#define BLOCK_DIM 32

/**
 * Computes the manhatten distance matrix between the query points and the reference points.
 *
 * @param ref          refence points stored in the global memory
 * @param ref_width    number of reference points
 * @param ref_pitch    pitch of the reference points array in number of column
 * @param query        query points stored in the global memory
 * @param query_width  number of query points
 * @param query_pitch  pitch of the query points array in number of columns
 * @param height       dimension of points = height of texture `ref` and of the array `query`
 * @param dist         array containing the query_width x ref_width computed distances
 * @param offset       the segment of the reference array to start at, if any
 */
__global__ void compute_distances_segment(float * ref,
                                  int     ref_width,
                                  int     ref_pitch,
                                  float * query,
                                  int     query_width,
                                  int     query_pitch,
                                  int     height,
                                  float * dist,
                                  int offset) {

    // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
    __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

    // Sub-matrix of A (begin, step, end) and Sub-matrix of B (begin, step)
    __shared__ int begin_A;
    __shared__ int begin_B;
    __shared__ int step_A;
    __shared__ int step_B;
    __shared__ int end_A;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Initializarion of the SSD for the current thread
    float ssd = 0.f;

    // Loop parameters
    begin_A = BLOCK_DIM * blockIdx.y + offset;  // account for segment offset
    begin_B = BLOCK_DIM * blockIdx.x;
    step_A  = BLOCK_DIM * ref_pitch;
    step_B  = BLOCK_DIM * query_pitch;
    end_A   = begin_A + (height - 1) * ref_pitch;

    // Conditions
    int cond0 = (begin_A + tx < ref_width);  // used to write in shared memory
    int cond1 = (begin_B + tx < query_width);  // used to write in shared memory & to computations and to write in output array 
    int cond2 = (begin_A + ty < ref_width);  // used to computations and to write in output matrix

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        if (a/ref_pitch + ty < height) {
            shared_A[ty][tx] = (cond0)? ref[a + ref_pitch * ty + tx] : 0;
            shared_B[ty][tx] = (cond1)? query[b + query_pitch * ty + tx] : 0;
        }
        else {
            shared_A[ty][tx] = 0;
            shared_B[ty][tx] = 0;
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
        if (cond2 && cond1) {
            for (int k = 0; k < BLOCK_DIM; ++k){
                float tmp = shared_A[k][ty] - shared_B[k][tx];
                ssd += fabsf(tmp);
            }
        }

        // Synchronize to make sure that the preceeding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory; each thread writes one element
    if (cond2 && cond1) {
        dist[ (begin_A + ty) * query_pitch + begin_B + tx ] = ssd;
    }

}


/**
 * For each reference point (i.e. each column) finds the k-th smallest distances
 * of the distance matrix and their respective indexes and gathers them at the top
 * of the 2 arrays.
 *
 * Since we only need to locate the k smallest distances, sorting the entire array
 * would not be very efficient if k is relatively small. Instead, we perform a
 * simple insertion sort by eventually inserting a given distance in the first
 * k values.
 *
 * @param dist         distance matrix
 * @param dist_pitch   pitch of the distance matrix given in number of columns
 * @param index        index matrix
 * @param index_pitch  pitch of the index matrix given in number of columns
 * @param width        width of the distance matrix and of the index matrix
 * @param height       height of the distance matrix
 * @param k            number of values to find
 */
__global__ void modified_insertion_sort(float * dist,
                                        int     dist_pitch,
                                        int *   index,
                                        int     index_pitch,
                                        int     width,
                                        int     height,
                                        int     k){

    // Column position
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    // Do nothing if we are out of bounds
    if (xIndex >= width) {
        return;
    }

    // Pointer shift
    float * p_dist  = dist  + xIndex;
    int *   p_index = index + xIndex;

    // Initialise the first index
    p_index[0] = 0;

    // Go through all points
    for (int i=0; i < height; ++i) {

        // Store current distance and associated index
        int   curr_index  = i;
        float curr_dist = p_dist[curr_index*dist_pitch];
        // Skip the current value if its index is >= k and if it's higher the k-th slready sorted smallest value
        if (curr_index >= k && curr_dist >= p_dist[(k-1)*dist_pitch]) {
            continue;
        }

        // Shift values (and indexes) higher that the current distance to the right
        int j = min(curr_index, k-1);
        while (j > 0 && p_dist[(j-1)*dist_pitch] > curr_dist) {
            p_dist[(j)*dist_pitch]  = p_dist[(j -1)*dist_pitch];
            p_index[(j)*index_pitch] = p_index[(j-1)*index_pitch];
            --j;
        }

        // Write the current distance and index at their position
        p_dist[(j)*dist_pitch]   = curr_dist;
        p_index[(j)*index_pitch] = curr_index;
    }

}

/**
 * Classify each query point by utilizing the k smallest distances stored in the ref and index arrays.
 *
 * @param index        index matrix   
 * @param ref          refence points stored in the global memory
 * @param ref_pitch    pitch of the reference points array in number of column
 * @param index_pitch  pitch of the index matrix given in number of columns
 * @param query_labels the result array
 * @param query_nb     number of query points
 * @param height       height of the ref matrix
 * @param k            number of values to find
 */
__global__ void majority_voting_kernel(const int* index, const float * ref, int ref_pitch, int* query_labels, int index_pitch, int query_nb, int height, int k) {
    // Get the query point index
    int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx < query_nb) {
        // Use shared memory to count votes
        extern __shared__ int shared_mem[];
        int* vote_count = shared_mem + threadIdx.x * k;

        // Initialize vote counts to zero
        for (int i = 0; i < k; i += 1) {
            vote_count[i] = 0;
        }
        __syncthreads();

        // Count votes for each label
        for (int j = 0; j < k; ++j) {
            int ref_index = index[query_idx + j*index_pitch];
            int label = ref[ref_index + (height)*ref_pitch];
            atomicAdd(&vote_count[label], 1);
        }
        __syncthreads();

        // Determine the majority label
        int majority_label = 0;
        int max_count = 0;
        for (int i = 0; i < k; ++i) {
            if (vote_count[i] > max_count) {
                max_count = vote_count[i];
                majority_label = i;
            }
        }
        // Assign the majority label to the query point
        query_labels[query_idx] = majority_label;
    }

}

void knn_cuda_free(float *ref_dev, float *query_dev, float *dist_dev, int *index_dev, int *query_labels_dev) {
    cudaFree(ref_dev);
    cudaFree(query_dev);
    cudaFree(dist_dev);
    cudaFree(index_dev);
    cudaFree(query_labels_dev);
}


bool knn_cuda_global(const float * ref,
                     int           ref_nb,
                     const float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     int *         query_labels, double &elapsed_time) {

    // Constants
    const unsigned int size_of_float = sizeof(float);
    const unsigned int size_of_int   = sizeof(int);

    // Return variables
    cudaError_t err0, err1, err2, err3, err4;

    // Check that we have at least one CUDA device 
    int nb_devices;
    err0 = cudaGetDeviceCount(&nb_devices);
    if (err0 != cudaSuccess || nb_devices == 0) {
        printf("ERROR: No CUDA device found\n");
        return false;
    }

    // Select the first CUDA device as default
    err0 = cudaSetDevice(0);
    if (err0 != cudaSuccess) {
        printf("ERROR: Cannot set the chosen CUDA device\n");
        return false;
    }

    // Allocate global memory
    float * ref_dev   = NULL;
    float * query_dev = NULL;
    float * dist_dev  = NULL;
    int   * index_dev = NULL;
    int   * query_labels_dev = NULL;
    size_t  ref_pitch_in_bytes;
    size_t  query_pitch_in_bytes;
    size_t  dist_pitch_in_bytes;
    size_t  index_pitch_in_bytes;
    printf("Attempting to allocate memory...\n");

    err0 = cudaMallocPitch((void**)&ref_dev,   &ref_pitch_in_bytes,   ref_nb   * size_of_float, dim);
    if (err0 != cudaSuccess) {
        printf("ERROR: Memory allocation error for ref_dev: %s\n", cudaGetErrorString(err0));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    err1 = cudaMallocPitch((void**)&query_dev, &query_pitch_in_bytes, query_nb * size_of_float, dim);
    if (err1 != cudaSuccess) {
        printf("ERROR: Memory allocation error for query_dev: %s\n", cudaGetErrorString(err1));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    err2 = cudaMallocPitch((void**)&dist_dev,  &dist_pitch_in_bytes,  query_nb * size_of_float, ref_nb);
    if (err2 != cudaSuccess) {
        printf("ERROR: Memory allocation error for dist_dev: %s\n", cudaGetErrorString(err2));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    err3 = cudaMallocPitch((void**)&index_dev, &index_pitch_in_bytes, query_nb * size_of_int,   k);
    if (err3 != cudaSuccess) {
        printf("ERROR: Memory allocation error for index_dev: %s\n", cudaGetErrorString(err3));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    err4 = cudaMalloc((void**)&query_labels_dev, query_nb * size_of_int);
    if (err4 != cudaSuccess) {
        printf("ERROR: Memory allocation error for query_labels_dev: %s\n", cudaGetErrorString(err4));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }


    // Deduce pitch values
    size_t ref_pitch   = ref_pitch_in_bytes   / size_of_float;
    size_t query_pitch = query_pitch_in_bytes / size_of_float;
    size_t dist_pitch  = dist_pitch_in_bytes  / size_of_float;
    size_t index_pitch = index_pitch_in_bytes / size_of_int;

    // Check pitch values
    if (query_pitch != dist_pitch || query_pitch != index_pitch) {
        printf("ERROR: Invalid pitch value\n");
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false; 
    }

    // Copy reference and query data from the host to the device
    err0 = cudaMemcpy2D(ref_dev,   ref_pitch_in_bytes,   ref,   ref_nb * size_of_float,   ref_nb * size_of_float,   dim, cudaMemcpyHostToDevice);
    if (err0 != cudaSuccess) {
        printf("ERROR: Memory allocation error for ref_dev: %s\n", cudaGetErrorString(err0));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    err1 = cudaMemcpy2D(query_dev, query_pitch_in_bytes, query, query_nb * size_of_float, query_nb * size_of_float, dim, cudaMemcpyHostToDevice);
    if (err1 != cudaSuccess) {
        printf("ERROR: Memory allocation error for query_dev: %s\n", cudaGetErrorString(err1));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));


    dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid0((query_nb + BLOCK_DIM - 1) / BLOCK_DIM, (ref_nb + BLOCK_DIM - 1) / BLOCK_DIM, 1);

    // segment the grid if we have a large amount of ref points
    int maxGridDimY = 65535;
    int numSegments = (grid0.y + maxGridDimY - 1) / maxGridDimY;
    for(int seg = 0; seg < numSegments; seg++) {
        int offset = seg * maxGridDimY * BLOCK_DIM;
        
        dim3 gridSegment(grid0.x,
            min(grid0.y - seg*maxGridDimY, maxGridDimY),
            1
        );
        compute_distances_segment<<<gridSegment, block0>>>(ref_dev, ref_nb, ref_pitch, query_dev, query_nb, query_pitch, dim, dist_dev, offset);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess) {
            printf("ERROR: Unable to execute distance kernel: %s\n", cudaGetErrorString(err));
            knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
            return false;
        }

    }

    int sortBlockDim = 256;
    dim3 block1(sortBlockDim, 1, 1);
    dim3 grid1((query_nb + sortBlockDim - 1) / sortBlockDim, 1, 1);

    // Sort the distances with their respective indexes in parallel
    modified_insertion_sort<<<grid1, block1>>>(dist_dev, dist_pitch, index_dev, index_pitch, query_nb, ref_nb, k);
    cudaDeviceSynchronize();
    cudaError_t sortError = cudaGetLastError();

    if (sortError != cudaSuccess) {
        printf("ERROR: Unable to execute sort kernel: %s\n",  cudaGetErrorString(sortError));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }


    // Perform majority voting and classification on the GPU
    int threads_per_block = 256;
    int num_blocks = (query_nb + threads_per_block - 1) / threads_per_block;
    int shared_memory_size = k * sizeof(int) * threads_per_block; // Adjust based on the number of classes

    majority_voting_kernel<<<num_blocks, threads_per_block, shared_memory_size>>>(index_dev, ref_dev, ref_pitch, query_labels_dev, index_pitch, query_nb, dim - 1, k);
    cudaDeviceSynchronize();
    cudaError_t votingError = cudaGetLastError();
    if (votingError != cudaSuccess) {
        printf("ERROR: Unable to execute voting kernel: %s\n", cudaGetErrorString(votingError));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    float msecTotal = 0.0f;
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    elapsed_time=((msecTotal)/1000);

    // Copy classifcation results from the device to the host
    err0 = cudaMemcpy(query_labels, query_labels_dev, query_nb * size_of_int, cudaMemcpyDeviceToHost);
    if (err0 != cudaSuccess) {
        printf("ERROR: Unable to copy results from device to host: %s\n", cudaGetErrorString(cudaGetLastError()));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false; 
    }

    // Memory clean-up
    knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);

    return true;
}
