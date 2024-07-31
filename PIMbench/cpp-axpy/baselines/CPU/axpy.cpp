/**
 * @file axpy.cpp
 * @brief Template for a Host Application Source File.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <stdint.h>
#include <iomanip>
#include <chrono>
#include <omp.h>

#include "../../../utilBaselines.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t vectorSize;
  int n_threads;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./program [options]"
          "\n"
          "\n    -t <T>    # of threads (default=8)"
          "\n    -i <I>    input size (default=8M elements)"
          "\n");
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.vectorSize = 65536;
  p.n_threads = 8;

  int opt;
  while ((opt = getopt(argc, argv, "i:t:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'i':
      p.vectorSize = atoi(optarg);
      break;
    case 't':
      p.n_threads = atoi(optarg);
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }

  assert(p.n_threads > 0 && "Invalid # of ranks!");
  return p;
}

/**
 * @brief cpu vector addition kernel
 */
void axpy(int32_t a, const std::vector<int32_t>& x, std::vector<int32_t>& y) 
{
  #pragma omp parallel for
  for (size_t i = 0; i < x.size(); ++i) 
  {
    y[i] = a * x[i] + y[i];
  }
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char **argv)
{
  struct Params p = input_params(argc, argv);

  uint64_t vectorSize = p.vectorSize;

  vector<int32_t> X;
  vector<int32_t> Y;

  getVector(vectorSize, X);
  getVector(vectorSize, Y);
  int32_t a = rand() % 5;

  auto start = std::chrono::high_resolution_clock::now();
  
  for (int32_t i = 0; i < WARMUP; i++)
  {
    axpy(a, X, Y);
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsedTime = (end - start)/WARMUP;
  cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;

  return 0;
}
