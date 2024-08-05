#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cub/device/device_segmented_sort.cuh>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
#define BLOCK_DIM 8

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
 * @param dist_pitch   pitch for the distance matrix
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
                                  int     dist_pitch,
                                  int offset) {

    // Declaration of the shared memory arrays As and Bs used to store the sub-matrix of A and B
    __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Initializarion of the SSD for the current thread
    float ssd = 0.f;

    // Loop parameters
    int begin_A = BLOCK_DIM * blockIdx.x + offset;  // account for segment offset
    int begin_B = BLOCK_DIM * blockIdx.y;
    int step_A  = BLOCK_DIM * ref_pitch;
    int step_B  = BLOCK_DIM * query_pitch;
    int end_A   = begin_A + (height-1) * ref_pitch;

    // Conditions
    int cond0 = (begin_A + tx < ref_width);  // used to write in shared memory
    int cond1 = (begin_B + ty < query_width);  // used to write in shared memory & to computations and to write in output array 
    //int cond2 = (begin_A + ty < ref_width);  // used to computations and to write in output matrix
    int cond2 = (cond0 && cond1);

    // Loop over all the sub-matrices of A and B required to compute the block sub-matrix
    for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {

        // Load the matrices from device memory to shared memory; each thread loads one element of each matrix
        if (a/ref_pitch + tx < height) {
            shared_A[ty][tx] = (cond0)? ref[a + ref_pitch * tx + ty] : 0;
            shared_B[ty][tx] = (cond1)? query[b + query_pitch * tx + ty] : 0;
        }
        else {
            shared_A[ty][tx] = 0;
            shared_B[ty][tx] = 0;
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Compute the difference between the two matrixes; each thread computes one element of the block sub-matrix
        if (cond2) {
            for (int k = 0; k < BLOCK_DIM; ++k){
                //float tmp = shared_A[k][tx] - shared_B[k][ty];
                float tmp = shared_A[tx][k] - shared_B[ty][k];
                ssd += fabsf(tmp);
            }
        }

        // Synchronize to make sure that the preceeding computation is done before loading two new sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory; each thread writes one element
    if (cond2) {
        int dist_index = (begin_B + ty) * dist_pitch + begin_A + tx;
        dist[ dist_index ] = ssd;
        //printf("At index %i we have this val %f\n", dist_index, ssd);
    }

}


__global__ void initializeIndex(int * index, size_t pitch, int width, int height, int offset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x + offset;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int* row = (int*)((char*)index + y * pitch);  // calculate row start
        row[x] = x;  // initialize with ascending value from 0 to width-1
        //printf("at index %i we have this value %i\n", x, x);
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
            int ref_index = index[query_idx * index_pitch + j];
            int label = ref[ref_index + height * ref_pitch];
            atomicAdd(&vote_count[label], 1);
        }
        __syncthreads();

        // Determine the majority label
        int majority_label = 0;
        int max_count = 0;
        for (int i = 0; i < k; ++i) {
            //printf("%i: here is current max_count %i and current vote_count %i\n", query_idx, max_count, vote_count[i]);
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
    cudaError_t err;

    // Check that we have at least one CUDA device 
    int nb_devices;
    err = cudaGetDeviceCount(&nb_devices);
    if (err != cudaSuccess || nb_devices == 0) {
        printf("ERROR: No CUDA device found\n");
        return false;
    }

    // Select the first CUDA device as default
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        printf("ERROR: Cannot set the chosen CUDA device\n");
        return false;
    }

    // Allocate global memory
    float * ref_dev   = NULL;
    float * query_dev = NULL;
    float * dist_dev  = NULL;
    float * sort_dist  = NULL;
    int   * index_dev = NULL;
    int   * sort_index = NULL;
    int   * query_labels_dev = NULL;
    int   * offsets = NULL;

    size_t  ref_pitch_in_bytes;
    size_t  query_pitch_in_bytes;
    size_t  dist_pitch_in_bytes;
    size_t  index_pitch_in_bytes;
    printf("Attempting to allocate memory...\n");

    err = cudaMallocPitch((void**)&ref_dev,   &ref_pitch_in_bytes,   ref_nb   * size_of_float, dim);
    if (err != cudaSuccess) {
        printf("ERROR: Memory allocation error for ref_dev: %s\n", cudaGetErrorString(err));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    err = cudaMallocPitch((void**)&query_dev, &query_pitch_in_bytes, query_nb * size_of_float, dim);
    if (err != cudaSuccess) {
        printf("ERROR: Memory allocation error for query_dev: %s\n", cudaGetErrorString(err));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    //err = cudaMallocPitch((void**)&dist_dev,  &dist_pitch_in_bytes,  query_nb * size_of_float, ref_nb);
    err = cudaMallocPitch((void**)&dist_dev,  &dist_pitch_in_bytes,  ref_nb * size_of_float, query_nb);
    if (err != cudaSuccess) {
        printf("ERROR: Memory allocation error for dist_dev: %s\n", cudaGetErrorString(err));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    err = cudaMallocPitch((void**)&sort_dist,  &dist_pitch_in_bytes,  ref_nb * size_of_float, query_nb);
    if (err != cudaSuccess) {
        printf("ERROR: Memory allocation error for sort_dist: %s\n", cudaGetErrorString(err));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    // err = cudaMallocPitch((void**)&index_dev, &index_pitch_in_bytes, query_nb * size_of_int, ref_nb);
    err = cudaMallocPitch((void**)&index_dev, &index_pitch_in_bytes, ref_nb * size_of_int, query_nb);
    if (err != cudaSuccess) {
        printf("ERROR: Memory allocation error for index_dev: %s\n", cudaGetErrorString(err));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    err = cudaMallocPitch((void**)&sort_index, &index_pitch_in_bytes, ref_nb * size_of_int, query_nb);
    if (err != cudaSuccess) {
        printf("ERROR: Memory allocation error for index_dev: %s\n", cudaGetErrorString(err));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    err = cudaMalloc((void**)&query_labels_dev, query_nb * size_of_int);
    if (err != cudaSuccess) {
        printf("ERROR: Memory allocation error for query_labels_dev: %s\n", cudaGetErrorString(err));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    err = cudaMalloc((void**)&offsets, (2*query_nb+1) * size_of_int);
    if (err != cudaSuccess) {
        printf("ERROR: Memory allocation error for offsets: %s\n", cudaGetErrorString(err));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    // Deduce pitch values
    size_t ref_pitch   = ref_pitch_in_bytes   / size_of_float;
    size_t query_pitch = query_pitch_in_bytes / size_of_float;
    size_t dist_pitch  = dist_pitch_in_bytes  / size_of_float;
    size_t index_pitch = index_pitch_in_bytes / size_of_int;

    // Check pitch values
    if (ref_pitch != dist_pitch || ref_pitch != index_pitch) {
        printf("ERROR: Invalid pitch value\n");
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false; 
    }

    // Copy reference and query data from the host to the device
    err = cudaMemcpy2D(ref_dev,   ref_pitch_in_bytes,   ref,   ref_nb * size_of_float,   ref_nb * size_of_float, dim, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("ERROR: Memory allocation error for ref_dev: %s\n", cudaGetErrorString(err));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    err = cudaMemcpy2D(query_dev, query_pitch_in_bytes, query, query_nb * size_of_float, query_nb * size_of_float, dim, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("ERROR: Memory allocation error for query_dev: %s\n", cudaGetErrorString(err));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    //printf("maxsize of ref array with pitch: %lu. where ref_pitch is equal to %lu and query_pitch %lu\n", (query_nb)*ref_pitch, ref_pitch, query_pitch);
    // intialize the offset array for sort on host and transfer to GPU memory
    int* offset_h = (int*) malloc((2*query_nb + 1)*size_of_int);
    int padLength = ref_pitch - ref_nb;
    offset_h[2*query_nb] = query_nb*ref_pitch - 1;
    offset_h[2*query_nb - 1] = query_nb*ref_pitch - padLength;
    offset_h[0] = 0;

    printf("here is ref and ref pitch %i, %i\n", ref_nb, ref_pitch);
    for (int i = 1; i < query_nb; i++) {
        int pitched_end = i*ref_pitch;
        offset_h[2*i - 1] = pitched_end - padLength;
        offset_h[2*i] = pitched_end;
    }
    for (int i = 0; i <= 2*query_nb; i++) {
        printf("here is offset: %i\n", offset_h[i]);
    }
    err = cudaMemcpy(offsets, offset_h, 2*query_nb + 1, cudaMemcpyHostToDevice);
    free(offset_h);
    if (err != cudaSuccess) {
        printf("ERROR: Memory intialization error for offset: %s\n", cudaGetErrorString(err));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false;
    }

    int num_sort_items_h = query_nb*ref_pitch - 1;
    int num_sort_segments_h = 2*query_nb;

    printf("here is sort items %i and sort segments %i\n", num_sort_items_h, num_sort_segments_h);


    // Determine temporary device storage requirements
    void     *d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    cub::DeviceSegmentedSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        dist_dev, sort_dist, index_dev, sort_index,
        num_sort_items_h, num_sort_segments_h, offsets, offsets + 1);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid0((ref_nb + BLOCK_DIM - 1) / BLOCK_DIM, (query_nb + BLOCK_DIM - 1) / BLOCK_DIM, 1);

    // segment the grid if we have a large amount of ref points
    int maxGridDimX = 65535;
    int numSegments = (grid0.x + maxGridDimX - 1) / maxGridDimX;
    for(int seg = 0; seg < numSegments; seg++) {
        int offset = seg * maxGridDimX * BLOCK_DIM;
        
        dim3 gridSegment(
            min(grid0.x - seg*maxGridDimX, maxGridDimX),
            grid0.y,
            1
        );
        compute_distances_segment<<<gridSegment, block0>>>(ref_dev, ref_nb, ref_pitch, query_dev, 
                                        query_nb, query_pitch, dim, dist_dev, dist_pitch, offset);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess) {
            printf("ERROR: Unable to execute distance kernel: %s\n", cudaGetErrorString(err));
            knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
            return false;
        }

        // Create values_in array for radix sort. This repersents the index into the distance matrix
        initializeIndex<<<gridSegment, block0>>>(index_dev, index_pitch, query_nb, ref_nb, offset);

        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("ERROR: Unable to execute intialize kernel: %s\n", cudaGetErrorString(err));
            knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
            return false;
        }
    }

    // radixSortSegmented: 1. create values_in array (indexes) 2. create offset array 2. determine temp storage requirements

    // Run sorting operation
    cub::DeviceSegmentedSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        dist_dev, sort_dist, index_dev, sort_index,
        num_sort_items_h, num_sort_segments_h, offsets, offsets + 1);

    cudaDeviceSynchronize();
    cudaError_t sortingError = cudaGetLastError();
    if (sortingError != cudaSuccess) {
        printf("ERROR: Unable to execute sorting kernel: %s\n", cudaGetErrorString(sortingError));
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
    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));

    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    // Record the stop event
    printf("Execution time of k nearest neighbor = %f ms\n", msecTotal);

    // Copy classifcation results from the device to the host
    err = cudaMemcpy(query_labels, query_labels_dev, query_nb * size_of_int, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("ERROR: Unable to copy results from device to host: %s\n", cudaGetErrorString(cudaGetLastError()));
        knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);
        return false; 
    }

    // Memory clean-up
    knn_cuda_free(ref_dev, query_dev, dist_dev, index_dev, query_labels_dev);

    return true;
}
