/**
 * @file copy.cpp
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
#include <iostream>
#include <vector>

#include "utilBaselines.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t vectorLength;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./copy.out [options]"
          "\n"
          "\n    -l    input size (default=2048 elements)"
          "\n");
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.vectorLength = 2048;

  int opt;
  while ((opt = getopt(argc, argv, "h:l:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'l':
      p.vectorLength = strtoull(optarg, NULL, 0);
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

/**
 * @brief cpu copy kernel
 */
void copy(const std::vector<int>& x, std::vector<int>& y) 
{
  #pragma omp parallel for
  for (size_t i = 0; i < x.size(); ++i) 
  {
    y[i] = x[i];
  }
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char **argv)
{
  struct Params p = input_params(argc, argv);

  uint64_t vectorSize = p.vectorLength;

  std::vector<int> X;
  std::vector<int> Y;

  getVector<int>(vectorSize, X);
  Y.resize(X.size());

  auto start = std::chrono::high_resolution_clock::now();
  
  for (int32_t i = 0; i < WARMUP; i++)
  {
    copy(X, Y);
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsedTime = (end - start)/WARMUP;
  std::cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << std::endl;

  return 0;
}