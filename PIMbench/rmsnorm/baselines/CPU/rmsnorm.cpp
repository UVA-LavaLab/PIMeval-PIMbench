/**
 * @file rmsnorm.cpp
 * @brief RMSNORM.
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <unistd.h>
#include <getopt.h>
#include <chrono>
#include <cblas.h>
#include <cmath>

#include "../../../../util/utilBaselines.h"

using namespace std;

// Global Vectors
vector<int> A;
vector<int> B;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t vectorLength;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./rmsnorm.out [options]"
          "\n"
          "\n    -l    vector size (default=128 elements)"
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
  p.vectorLength = 128;

  int opt;
  while ((opt = getopt(argc, argv, ":l:h:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
    case 'l':
      p.vectorLength = stoull(optarg);
      break;
    default:
      cerr << "\nUnrecognized option: " << opt << "\n";
      usage();
      exit(1);
    }
  }
  return p;
}

void rmsnorm(uint64_t vectorLength, std::vector<int> &srcVector, std::vector<int> &dst)
{
uint32_t sum_sq = 0;
for (size_t i = 0; i < vectorLength; i++) 
{
  sum_sq += (uint32_t)(srcVector[i] * srcVector[i]); // Prevent overflow
}
uint32_t mean_sq = sum_sq / vectorLength;
uint32_t rms = sqrt(mean_sq+1); 
for (size_t i = 0; i < vectorLength; i++) 
{
  dst[i] = srcVector[i] / (rms + 1);  // Prevent division by zero
}
}

/**
 * @brief Main function.
 */
int main(int argc, char **argv)
{
  // Parse input parameters
  Params params = parseParams(argc, argv);
  uint64_t vectorLength = params.vectorLength;

  // Initialize vectors
  getVector(vectorLength, A);
  B.resize(vectorLength);
  std::cout << "Done initialization." << std::endl;

  auto start = chrono::high_resolution_clock::now();

  for (int32_t i = 0; i < WARMUP; i++)
  {
    rmsnorm(vectorLength, A, B);
  }

  auto end = chrono::high_resolution_clock::now();

  chrono::duration<double, milli> elapsedTime = (end - start) / WARMUP;
  cout << "Duration: " << fixed << setprecision(3) << elapsedTime.count() << " ms." << endl;

  return 0;
}
