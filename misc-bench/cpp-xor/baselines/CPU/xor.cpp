/**
 * @file xor.cpp
 * @brief Template for a Host Application Source File.
 *
 */

#include <chrono>
#include <unistd.h>
#include <omp.h>

#include "../../../../PIMbench/utilBaselines.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t dataSize;
  char *inputFile;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./xor.out [options]"
          "\n"
          "\n    -l    vector length (default=2048 elements)"
          "\n    -i    input file containing two vectors (default=generates vectors with random numbers)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.dataSize = 2048;
  p.inputFile = nullptr;

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
  std::vector<int> src1, src2, dst;
  uint64_t vectorLength = params.dataSize;
  
  if (params.inputFile == nullptr)
  {
    src1.resize(vectorLength);
    src2.resize(vectorLength);
    dst.resize(vectorLength);
    
    getVector(vectorLength, src1);
    getVector(vectorLength, src2);
  }
  else
  {
    std::cerr << "Reading from the input file is not implemented yet for the input matrix" << std::endl;
    return 1;
  }

  printf("Performing XOR CPU baseline with %lu data points\n", vectorLength);
  
  // Start timing
  auto start = std::chrono::high_resolution_clock::now();

  #pragma omp parallel for
  for (uint64_t i = 0; i < vectorLength; ++i)
  {
    dst[i] = src1[i] ^ src2[i];
  }

  // End Timing
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsedTime = end - start;
  std::cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << std::endl;
   
  return 0;
}
