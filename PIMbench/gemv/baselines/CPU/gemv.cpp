/**
 * @file gemv.cpp
 * @brief GEMV.
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <unistd.h>
#include <getopt.h>
#include <chrono>
#include <cblas.h>

#include "utilBaselines.h"

using namespace std;

// Global Vectors
vector<float> A;
vector<float> B;
vector<float> C;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t row, col;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./gemv.out [options]"
          "\n"
          "\n    -r    row size (default=16384 elements)"
          "\n    -c    column size (default=16384 elements)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

/**
 * @brief Parses command line input parameters
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return Parsed parameters
 */
struct Params parseParams(int argc, char **argv)
{
  struct Params p;
  p.row = 16384;
  p.col = 16384;

  int opt;
  while ((opt = getopt(argc, argv, ":r:c:h:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
    case 'r':
      p.row = stoull(optarg);
      break;
    case 'c':
      p.col = stoull(optarg);
      break;
    default:
      cerr << "\nUnrecognized option: " << opt << "\n";
      usage();
      exit(1);
    }
  }
  return p;
}

/**
 * @brief Main function.
 */
int main(int argc, char **argv)
{
  // Parse input parameters
  Params params = parseParams(argc, argv);
  uint64_t rows = params.row, cols = params.col;

  // Initialize vectors
  getVector(rows*cols, A);
  getVector(cols, B);
  float alpha = 1.0;
  float beta = 1.0f;
  C.resize(rows);
  std::cout << "Done initialization." << std::endl;

  auto start = chrono::high_resolution_clock::now();

  for (int32_t i = 0; i < WARMUP; i++)
  {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, alpha, A.data(), cols, B.data(), 1, beta, C.data(), 1);
  }

  auto end = chrono::high_resolution_clock::now();

  chrono::duration<double, milli> elapsedTime = (end - start) / WARMUP;
  cout << "Duration: " << fixed << setprecision(3) << elapsedTime.count() << " ms." << endl;

  return 0;
}