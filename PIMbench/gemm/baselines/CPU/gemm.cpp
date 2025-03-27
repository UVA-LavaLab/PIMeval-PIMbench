/**
 * @file gemm.cpp
 * @brief Template for a Host Application Source File.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <stdint.h>
#include <bits/stdc++.h>
#include <cblas.h>
#include <chrono>
#include <omp.h>
#include <iomanip>

#include "utilBaselines.h"

vector<double> A;
vector<double> B;
vector<double> C;

#define BLOCK_SIZE 64

// Params ---------------------------------------------------------------------
typedef struct Params
{
    uint64_t row, columnA, columnB;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./gemm.out [options]"
          "\n"
          "\n    -r <R>    row size"
          "\n    -c <C>    MatA column size"
          "\n    -d <C>    MatB column size"
          "\n");
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.row = 1024;
  p.columnA = 1024;
  p.columnB = 1024;

  int opt;
  while ((opt = getopt(argc, argv, ":r:c:d:")) >= 0)
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
      p.columnA = atoll(optarg);
      break;
    case 'd':
      p.columnB = atoll(optarg);
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

void matrix_matrix_multiplication(uint64_t row, uint64_t columnA, uint64_t columnB) 
{
  const double alpha = 1.0;
  const double beta = 0.0;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row, columnB, columnA, alpha, A.data(), columnA, B.data(), columnB, beta, C.data(), columnB);
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char **argv)
{
  struct Params p = input_params(argc, argv);

  getVector(p.row * p.columnA, A);
  getVector(p.columnA * p.columnB, B);
  C.resize(p.row * p.columnB);

  // Start timing  
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < WARMUP; i++)
  {
    matrix_matrix_multiplication(p.row, p.columnA, p.columnB);
  }

  // End timing
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsedTime = (end - start)/WARMUP;
  cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;

  return 0;
}
