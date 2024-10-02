/**
 * @file bitmap.cpp
 * @brief Template for a Host Application Source File.
 *
 */

#include <chrono>
#include <unistd.h>
#include <omp.h>
#include <random>

#include "../../../../PIMbench/utilBaselines.h"

// Static definition, should be made dynamic in future work
#define numBitmapIndices 8

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t dataSize;
  char *inputFile;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./bitmap.out [options]"
          "\n"
          "\n    -l    number of data entries (default=2048 elements)"
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
  std::vector<uint8_t> database;
  uint64_t numDatabaseEntries = params.dataSize;
  std::vector<uint8_t> validEntries;

  std::vector<std::vector<uint8_t>> result(numBitmapIndices, vector<uint8_t> (numDatabaseEntries));
  
  if (params.inputFile == nullptr)
  {
    database.resize(numDatabaseEntries);

    // Assuming 8 unique bitmap indicies, no database entries for 0x00
    validEntries = {
      0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80
    };
    for (uint64_t i = 0; i < numDatabaseEntries; ++i)
    {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> dis(0, validEntries.size() - 1);
      
      int idx = dis(gen);

      database[i] = validEntries[idx];
    }
  }
  else
  {
    std::cerr << "Reading from the input file is not implemented yet for bitmap" << std::endl;
    return 1;
  }

  printf("Performing bitmap CPU baseline with %lu data points and 8 unique bitmap indices\n", numDatabaseEntries);
  
  // Start timing
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < numBitmapIndices; ++i) 
  {   
    #pragma omp parallel for 
    for (uint64_t j = 0; j < numDatabaseEntries; ++j)
    {
      result[i][j] = (database[j] == validEntries[validEntries.size() - i - 1]);
    }
  }

  // End Timing
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsedTime = end - start;
  std::cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << std::endl;
   
  return 0;
}
