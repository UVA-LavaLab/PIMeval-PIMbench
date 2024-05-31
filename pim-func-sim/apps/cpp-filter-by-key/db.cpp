// Test: C++ version of filter by key
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../util.h"
#include "libpimsim.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t dbSize, numberOfKeys;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./db [options]"
          "\n"
          "\n    -k    number of keys to search (default=16 elements)"
          "\n    -d    database size (default=65536 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing two vectors (default=generates vector with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.dbSize = 65536;
  p.numberOfKeys = 15;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:k:d:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'k':
      p.numberOfKeys = strtoull(optarg, NULL, 0);
      break;
    case 'd':
      p.dbSize = strtoull(optarg, NULL, 0);
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

void keyValueSearchCPU(std::vector<int> &dbVector, int key, std::vector<int> &deviceBitMap)
{
  std::vector<int> hostBitMap(dbVector.size(), 0);
#pragma omp parallel for
  for (int i = 0; i < dbVector.size(); ++i)
  {
    if (key == dbVector[i])
      hostBitMap[i] = 1;
  }
  if (hostBitMap != deviceBitMap)
  {
    std::cout << "Incorrect result" << endl;
    exit(1);
  }
}

void keyValueSearch(std::vector<int> &dbVector, std::vector<int> &keyList, bool shouldVerify)
{
  unsigned bitsPerElement = sizeof(int) * 8;
  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, dbVector.size(), bitsPerElement, PIM_INT32);
  if (srcObj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  PimObjId srcObj2 = pimAllocAssociated(bitsPerElement, srcObj1, PIM_INT32);
  if (srcObj2 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice((void *)dbVector.data(), srcObj1);

  for (int keyToSearch : keyList)
  {
    std::vector<int> bitMap(dbVector.size());
    status = pimBroadcast(srcObj2, keyToSearch);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    status = pimEQ(srcObj1, srcObj2, srcObj2);

    status = pimCopyDeviceToHost(srcObj2, (void *)dbVector.data());
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
    }
    if (shouldVerify)
    {
      keyValueSearchCPU(dbVector, keyToSearch, bitMap);
    }
  }
}

void getKeysToMatch(int k, std::vector<int> &db, std::vector<int> &searchList)
{
  searchList.resize(k);

  for (int i = 0; i < k; i++)
  {
    int idx = rand() % db.size();
    searchList[i] = db[idx];
  }
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "DB element size: " << params.dbSize << ". Keys to search: " << params.numberOfKeys << "\n";

  std::vector<int> db, valueList, searchList;
  if (params.inputFile == nullptr)
  {
    getVector(params.dbSize, db);
    getKeysToMatch(params.numberOfKeys, db, searchList);
  }
  else
  {
    // TODO: Read from files
  }

  if (!createDevice(params.configFile))
    return 1;

  // TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iteration.
  keyValueSearch(db, searchList, params.shouldVerify);

  pimShowStats();

  return 0;
}
