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

#include "../../../utilBaselines.h"

using namespace std;

// Global Vectors
vector<vector<float>> Abatch;
vector<vector<float>> Bbatch;
vector<vector<float>> Cbatch;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t row, col, batch;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./gemv.out [options]"
          "\n"
          "\n    -r    row size (default=16384 elements)"
          "\n    -c    column size (default=16384 elements)"
          "\n    -b    batch size (default=1)"
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
  p.batch = 1;

  int opt;
  while ((opt = getopt(argc, argv, ":r:c:b:v:h:")) >= 0)
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
    case 'b':
      p.batch = stoull(optarg);
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
  uint64_t batch = params.batch;

  // Constant parameters used in CUBlas library
  float alpha = 1.0;
  float beta = 1.0f;
  
  // Resize the batch vectors
  Abatch.resize(params.batch);
  Bbatch.resize(params.batch);
  Cbatch.resize(params.batch);

  // Initialize vectors
  for (uint64_t i = 0; i < batch; i++) 
  {
    getVector(rows*cols, Abatch[i]);
    getVector(cols, Bbatch[i]);
    Cbatch[i].resize(rows);
  }
  std::cout << "Done initialization." << std::endl;

  auto start = chrono::high_resolution_clock::now();

  for (int32_t i = 0; i < WARMUP; i++)
  {
    for (uint64_t i = 0; i < batch; i++) 
    {
      cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, alpha, Abatch[i].data(), cols, Bbatch[i].data(), 1, beta, Cbatch[i].data(), 1);
    }
  }

  auto end = chrono::high_resolution_clock::now();

  chrono::duration<double, milli> elapsedTime = (end - start) / WARMUP;
  cout << "Duration: " << fixed << setprecision(3) << elapsedTime.count() << " ms." << endl;

  return 0;
}
