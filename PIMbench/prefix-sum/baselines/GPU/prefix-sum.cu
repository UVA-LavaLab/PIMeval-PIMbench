/* File:     axpy.cu
 * Purpose:  Implement axpy on a gpu using cuda
 *
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cub/cub.cuh>
#include <nvml.h>

#include "utilBaselines.h"

using namespace std;

vector<int32_t> A;
vector<int32_t> C;

#define TOLERANCE	200.0f

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t vector_size;
} Params;

void usage()
{
  fprintf(stderr,
          "\n Usage:  ./prefix-sum.out [options]"
          "\n"
          "\n    -i    vector size (default=65536)"
          "\n");
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.vector_size = 65536;

  int opt;
  while ((opt = getopt(argc, argv, ":h:i:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'i':
      p.vector_size = atoll(optarg);
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

  uint64_t vector_size = p.vector_size;
  getVector(vector_size, A);
  int *d_in, *d_out;
  cudaError_t errorCode;

  errorCode = cudaMalloc(&d_in, sizeof(int) * vector_size);
  if (errorCode != cudaSuccess)
  {
      cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
      exit(1);
  }
  errorCode = cudaMalloc(&d_out, sizeof(int) * vector_size);
  {
      cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
      exit(1);
  }

  // Determine temporary device storage requirements
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out,
                                vector_size);

  errorCode = cudaMalloc(&d_temp_storage, temp_storage_bytes);
  {
      cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
      exit(1);
  }

  errorCode = cudaMemcpy(d_in, A.data(), sizeof(int) * vector_size, cudaMemcpyHostToDevice);
  {
      cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
      exit(1);
  }

  auto [timeElapsed, avgPower, energy] = measureCUDAPowerAndElapsedTime([&]() {
    /* Kernel Call */
    errorCode = cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out,
                                vector_size);
  });
  // Check for kernel launch errors
  errorCode = cudaGetLastError();
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  printf("Execution time for AXPY = %f ms\n", timeElapsed);
  printf("Average Power = %f mW\n", avgPower);
  printf("Energy Consumption = %f mJ\n", energy);

  vector<int32_t> C(vector_size);
  errorCode = cudaMemcpy(C.data(), d_out, vector_size * sizeof(int32_t), cudaMemcpyDeviceToHost);
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error Copy from device to host: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  cout.precision(0);
  for (size_t i = 0; i < A.size(); ++i)
  {
    int32_t sum = a * B[i] + A[i];
    if (abs(C[i] - sum) > TOLERANCE)
    {
      cout << fixed << "AXPY failed at index: " << i << "\t" << C[i] << "\t" << sum << endl;
      break;
    }
  }

  /* Free memory */
  cudaFree(d_in);
  cudaFree(d_out);

  return 0;
} /* main */
