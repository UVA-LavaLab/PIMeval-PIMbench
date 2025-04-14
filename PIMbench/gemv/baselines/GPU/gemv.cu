/* File:     gemv.cu
 * Purpose:  Implement matrix vector multiplication on a gpu using cuda
 *
 */

#include <chrono>
#include <cublas_v2.h>
#include <iostream>
#include <math.h>
#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <unistd.h>

#include "utilBaselines.h"

#define TOLERANCE 200.0f

using namespace std;

vector<float> A;
vector<float> B;
vector<float> C;

// Params ---------------------------------------------------------------------
typedef struct Params {
  uint64_t row, column;
  bool shouldVerify;
} Params;

void usage() {
  fprintf(
      stderr,
      "\nUsage:  ./gemv.out [options]"
      "\n"
      "\n    -r    row size (default=16384)"
      "\n    -c    column size (default=16384)"
      "\n    -v    t = verifies PIM output with host output. (default=false)"
      "\n");
}

struct Params input_params(int argc, char **argv) {
  struct Params p;
  p.row = 16384;
  p.column = 16384;

  int opt;
  while ((opt = getopt(argc, argv, ":r:c:h:v:")) >= 0) {
    switch (opt) {
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

int main(int argc, char *argv[]) {
  struct Params p = input_params(argc, argv);

  uint64_t row = p.row, col = p.column;
  getVector(row * col, A);
  getVector(col, B);
  C.resize(row);

  std::cout << "Starting GEMV For: " << row << " X " << col << std::endl;

  float *x, *y, *z;

  cudaError_t errorCode;

  errorCode = cudaMalloc((void **)&x, row * col * sizeof(int32_t));
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMalloc((void **)&y, col * sizeof(int32_t));
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMalloc((void **)&z, row * sizeof(int32_t));
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  errorCode = cudaMemcpy(x, A.data(), row * col * sizeof(float),
                         cudaMemcpyHostToDevice);
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  errorCode =
      cudaMemcpy(y, B.data(), col * sizeof(float), cudaMemcpyHostToDevice);
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  const float alpha = 1.0;
  const float beta = 0.0;
  cublasHandle_t handle;
  cublasStatus_t status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "CUBLAS initialization failed\n";
    exit(1);
  }

  auto [timeElapsed, avgPower, energy] = measureCUDAPowerAndElapsedTime([&]() {
    /* Kernel Call */
    status = cublasSgemv(handle, CUBLAS_OP_N, row, col, &alpha, x, row, y, 1,
                       &beta, z, 1);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
      std::cerr << "CUBLAS SGEMV failed\n";
      exit(1);
    }
  });
  // Check for kernel launch errors
  errorCode = cudaGetLastError();
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  printf("Execution time of gemv = %f ms\n", timeElapsed);
  printf("Average Power = %f mW\n", avgPower);
  printf("Energy Consumption = %f mJ\n", energy);

  errorCode = cudaMemcpy(C.data(), z, row * sizeof(int32_t), cudaMemcpyDeviceToHost);
  if (errorCode != cudaSuccess) {
    cerr << "Cuda Error Copy from host to device: "
         << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  if (p.shouldVerify) {
    cout.precision(0);
    for (int i = 0; i < row; ++i) {
      int32_t sum = 0;
      for (int j = 0; j < col; ++j) {
        sum += A[i + j * row] * B[j];
      }
      if (abs(C[i] - sum) > TOLERANCE) {
        cout << fixed << "Multiplication failed at index: " << i << "\t" << C[i]
             << "\t" << sum << endl;
        break;
      }
    }
    cout << "All correct!" << endl;
  }

  /* Free memory */
  cublasDestroy(handle);
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);

  return 0;
} /* main */
