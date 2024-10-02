// Test: C++ version of bitmap
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <unistd.h>
#include <cassert>
#include <random>
#include <cstdint>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../../util.h"
#include "libpimeval.h"

using namespace std;

// Static definition, should be made dynamic in future work
#define numBitmapIndices 8

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t dataSize;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./bitmap.out [options]"
          "\n"
          "\n    -l    number of data entries (default=2048 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing two vectors (default=generates vectors with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.dataSize = 2048;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:l:c:i:v:")) >= 0)
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
    case 'c':
      p.configFile = optarg;
      break;
    case 'i':
      p.inputFile = optarg;
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

void bitmap(const uint64_t numDatabaseEntries, const std::vector<uint8_t> database, const std::vector<uint8_t> indicesChecker, std::vector<std::vector<uint8_t>> &result)
{
  PimObjId databaseObj = pimAlloc(PIM_ALLOC_AUTO, numDatabaseEntries, PIM_UINT8);
  assert(databaseObj != -1);
  PimObjId indicesCheckerObj = pimAllocAssociated(databaseObj, PIM_UINT8);
  assert(indicesCheckerObj != -1);
  PimObjId tempObj = pimAllocAssociated(databaseObj, PIM_UINT8);
  assert(tempObj != -1);

  PimStatus status = pimCopyHostToDevice((void *) database.data(), databaseObj);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void *) indicesChecker.data(), indicesCheckerObj);
  assert(status == PIM_OK);

  for (uint64_t i = 0; i < numBitmapIndices; ++i)
  {
    status = pimEQ(databaseObj, indicesCheckerObj, tempObj);
    assert(status == PIM_OK);
  
    status = pimShiftBitsRight(indicesCheckerObj, indicesCheckerObj, 1);
    assert(status == PIM_OK);

    status = pimCopyDeviceToHost(tempObj, (void *) result[i].data());
    assert(status == PIM_OK);
  }

  pimFree(databaseObj);
  pimFree(indicesCheckerObj);
  pimFree(tempObj);
}

int main(int argc, char *argv[])
{
  //TODO: Add dynamic support for number of unique bitmap indices (i.e. UINT4 when #indices=4, UINT16 when #indices=16, etc.)
  struct Params params = getInputParams(argc, argv);
  std::vector<uint8_t> database, indicesChecker;
  uint64_t numDatabaseEntries = params.dataSize;
  std::vector<uint8_t> validEntries;

  std::vector<std::vector<uint8_t>> result(numBitmapIndices, vector<uint8_t> (numDatabaseEntries));
  
  if (params.inputFile == nullptr)
  {
    database.resize(numDatabaseEntries);
    indicesChecker.resize(numDatabaseEntries);

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
      indicesChecker[i] = 0x80;
    }
  }
  else
  {
    std::cerr << "Reading from the input file is not implemented yet for bitmap" << std::endl;
    return 1;
  }

  printf("Performing bitmap with %llu data points and 8 unique bitmap indices\n", numDatabaseEntries);

  if (!createDevice(params.configFile))
  {
    return 1;
  }

  bitmap(numDatabaseEntries, database, indicesChecker, result);

  if (params.shouldVerify)
  {  
    int errorFlag = 0;
    std::vector<std::vector<uint8_t>> baselineResult(numBitmapIndices, vector<uint8_t> (numDatabaseEntries));

    for (int i = 0; i < numBitmapIndices; ++i) 
    {   
      #pragma omp parallel for 
      for (uint64_t j = 0; j < numDatabaseEntries; ++j)
      {
        baselineResult[i][j] = (database[j] == validEntries[validEntries.size() - i - 1]);
        if (baselineResult[i][j] != result[i][j]) 
        {
          std::cout << "Wrong answer at index [" << i << "," << j << "] | Wrong PIM answer = " << static_cast<int> (result[i][j]) << " (Baseline expected = " << static_cast<int> (baselineResult[i][j]) << ")" << std::endl;
          errorFlag = 1;
        }
      }
    }
    if (!errorFlag)
    {
      std::cout << "Correct!" << std::endl;
    }
  }

  pimShowStats();

  return 0;
}
