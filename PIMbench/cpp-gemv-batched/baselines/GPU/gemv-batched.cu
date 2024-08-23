/* File:     gemv.cu
 * Purpose:  Implement matrix vector multiplication on a gpu using cuda
 *
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cublas_v2.h>

#include "../../../utilBaselines.h"

#define TOLERANCE 200.0f

using namespace std;

// Global Vectors
vector<vector<float>> Abatch;
vector<vector<float>> Bbatch;
vector<vector<float>> Cbatch;


// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t row, column, batch;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./gemv.out [options]"
          "\n"
          "\n    -r    row size (default=16384)"
          "\n    -c    column size (default=16384)"
          "\n    -b    batch size (default=1)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.row = 16384;
  p.column = 16384;
  p.batch = 1;

  int opt;
  while ((opt = getopt(argc, argv, ":r:c:b:h:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'r':
      p.row = atoll(optarg);
      break;
    case 'c':
      p.column = atoll(optarg);
      break;
    case 'b':
      p.batch = atoll(optarg);
      break;
    case 'v':
      p.shouldVerify = (*optarg == 't') ? true : false;
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}
int main(int argc, char *argv[])
{
    struct Params p = input_params(argc, argv);

    uint64_t row = p.row, col = p.column, batch = p.batch;

    // Resize the batch vectors
    Abatch.resize(batch);
    Bbatch.resize(batch);
    Cbatch.resize(batch);

    // Initialize vectors
    for (uint64_t i = 0; i < batch; i++)
    {
        getVector(row * col, Abatch[i]);
        getVector(col, Bbatch[i]);
        Cbatch[i].resize(row);
    }
    std::cout << "Done initialization." << std::endl;

    float *x, *y, *z;

    cudaError_t errorCode;

    // Allocate memory for each batch
    errorCode = cudaMalloc((void **)&x, batch * row * col * sizeof(float));
    if (errorCode != cudaSuccess)
    {
        std::cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }
    errorCode = cudaMalloc((void **)&y, batch * col * sizeof(float));
    if (errorCode != cudaSuccess)
    {
        std::cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }
    errorCode = cudaMalloc((void **)&z, batch * row * sizeof(float));
    if (errorCode != cudaSuccess)
    {
        std::cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "CUBLAS initialization failed\n";
        exit(1);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float timeElapsed = 0;

    // Start timer
    cudaEventRecord(start, 0);

    // Process each batch
    for (uint64_t i = 0; i < batch; i++)
    {
        // Copy data to device for each batch
        errorCode = cudaMemcpy(x + i * row * col, Abatch[i].data(), row * col * sizeof(float), cudaMemcpyHostToDevice);
        if (errorCode != cudaSuccess)
        {
            std::cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
            exit(1);
        }

        errorCode = cudaMemcpy(y + i * col, Bbatch[i].data(), col * sizeof(float), cudaMemcpyHostToDevice);
        if (errorCode != cudaSuccess)
        {
            std::cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
            exit(1);
        }

        // GEMV kernel for each batch
        const float alpha = 1.0f;
        const float beta = 0.0f;
        status = cublasSgemv(handle, CUBLAS_OP_N, row, col, &alpha, x + i * row * col, row, y + i * col, 1, &beta, z + i * row, 1);
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr << "CUBLAS SGEMV failed\n";
            exit(1);
        }

        // Copy result back to host for each batch
        errorCode = cudaMemcpy(Cbatch[i].data(), z + i * row, row * sizeof(float), cudaMemcpyDeviceToHost);
        if (errorCode != cudaSuccess)
        {
            std::cerr << "Cuda Error Copy from host to device: " << cudaGetErrorString(errorCode) << "\n";
            exit(1);
        }

        if (p.shouldVerify)
        {
            std::cout.precision(0);
            for (int j = 0; j < row; ++j)
            {
                int32_t sum = 0;
                for (int k = 0; k < col; ++k)
                {
                    sum += Abatch[i][j + k * row] * Bbatch[i][k];
                }
                if (abs(Cbatch[i][j] - sum) > TOLERANCE)
                {
                    std::cout << std::fixed << "Multiplication failed at batch: " << i << ", index: " << j << "\t" << Cbatch[i][j] << "\t" << sum << std::endl;
                    break;
                }
            }
            std::cout << "Batch " << i << " correct!" << std::endl;
        }
    }

    // End timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeElapsed, start, stop);

    printf("Total execution time for %lu batches = %f ms\n", batch, timeElapsed);

    // Free memory
    cublasDestroy(handle);
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
}

