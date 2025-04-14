/**
 * @file lr.cpp
 *
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <iomanip>

#include "utilBaselines.h"

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t dataSize;
  std::string inputFile;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./lr.out [options]"
          "\n"
          "\n    -l    input size (default=2048 elements)"
          "\n    -i    input file containing 2D matrix (default=generates matrix with random numbers)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.dataSize = 2048;
  p.inputFile = "";

  int opt;
  while ((opt = getopt(argc, argv, "h:l:i:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'l':
      p.dataSize = strtoull(optarg, NULL, 0);
      break;
    case 'i':
      p.inputFile = optarg;
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
  struct Params params = getInputParams(argc, argv);

  if (params.inputFile != "")
  {
    std::cout << "Need work reading in files" << std::endl;
    return 1;
  }

  uint64_t n = params.dataSize;

  vector<int32_t> dataPointsX, dataPointsY;
  getVector(n, dataPointsX);
  getVector(n, dataPointsY);
  cout << "Done initializing data\n";

  double slope, intercept;

  auto start = std::chrono::high_resolution_clock::now();
  for (int32_t w = 0; w < WARMUP; w++) 
  {
    double SX = 0, SY = 0, SXX = 0, SXY = 0;
    
    #pragma omp parallel for simd reduction(+ : SX, SXX, SY, SXY)
    for (uint64_t i = 0; i < n; i++)
    {
      SX += dataPointsX[i];
      SXX += dataPointsX[i] * dataPointsX[i];
      SY += dataPointsY[i];
      SXY += dataPointsX[i] * dataPointsX[i];
    }
    // Calculate slope and intercept
    slope = (n * SXY - SX * SY) / (n * SXX - SX * SX);
    intercept = (SY - slope * SX) / n;
  }
  
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsedTime = (end - start)/WARMUP;
  cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;

  std::cout << "Slope = " << slope << ", Intercept = " << intercept << std::endl;

  return 0;
} /* main */
