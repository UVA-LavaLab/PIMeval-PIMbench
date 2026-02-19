// Test: C++ version of breadth-first search
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#include <unordered_map>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "util.h"
#include "libpimeval.h"

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t vectorLength;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./bfs.out [options]"
          "\n"
          "\n    -l    input size (default=2048 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing two vectors (default=generates vector with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.vectorLength = 2048;
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
      p.vectorLength = strtoull(optarg, NULL, 0);
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

void pinVerticesToCores(uint64_t numVertices, PimDeviceProperties &deviceProps)
{

}

void vectorAddition(uint64_t vectorLength, std::vector<int> &src1, std::vector<int> &src2, std::vector<int> &dst)
{
  PimDeviceProperties deviceProps;
  PimStatus status = pimGetDeviceProperties(&deviceProps);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  // Let's assume vector Length = #vertices
  uint64_t verticesPerCore = std::ceil(vectorLength * 1.0 / deviceProps.numPIMCores);

  // what should be the length of the object containing vertices belonging to each core
  uint64_t elementsPerRow = deviceProps.numColPerSubarray / (sizeof(int) * 8);
  std::cout << "Num Columns per Subarray: " << deviceProps.numColPerSubarray << ", Bits per element: " << sizeof(int) * 8 << ", Elements per row: " << elementsPerRow << "\n";

  // initialize vertex vector with -1; -1 indicates padded vertices
  std::vector<int> vertexVector(elementsPerRow * deviceProps.numPIMCores, -1);
  std::cout << "vectorLength: " << vectorLength << ", verticesPerCore: " << verticesPerCore << ", elementsPerRow: " << elementsPerRow << "\n";
  for (unsigned coreId = 0; coreId < deviceProps.numPIMCores; ++coreId)
  {
    int startVertex = coreId * verticesPerCore;
    int endVertex = std::min(startVertex + verticesPerCore, vectorLength);
    vertexVector[coreId * elementsPerRow] = startVertex;
    vertexVector[coreId * elementsPerRow + 1] = endVertex;
  }

  std::vector<uint8_t> startMask(vertexVector.size(), 0);
  std::vector<uint8_t> endMask(vertexVector.size(), 0);
  for (unsigned coreId = 0; coreId < deviceProps.numPIMCores; ++coreId) {
    uint64_t base = (uint64_t)coreId * elementsPerRow;
    startMask[base + 0] = 1;
    endMask[base + 1] = 1;
  }

  std::vector<int> frontierVector(elementsPerRow * deviceProps.numPIMCores, 0);

  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, elementsPerRow * deviceProps.numPIMCores, PIM_INT32);
  if (srcObj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimCopyHostToDevice((void *)vertexVector.data(), srcObj1);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId srcObj2 = pimAllocAssociated(srcObj1, PIM_BOOL);
  if (srcObj2 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId srcObj3 = pimAllocAssociated(srcObj1, PIM_BOOL);
  if (srcObj3 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId stMask = pimAllocAssociated(srcObj1, PIM_BOOL);
  if (stMask == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId enMask = pimAllocAssociated(srcObj1, PIM_BOOL);
  if (enMask == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimCopyHostToDevice((void *)startMask.data(), stMask);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  
  status = pimCopyHostToDevice((void *)endMask.data(), enMask);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimLTScalar(srcObj1, srcObj2, 401, 0, 1, PIM_LOCAL);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
  }

  status = pimGTScalar(srcObj1, srcObj3, 400, 1, 2, PIM_LOCAL);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
  }

  status = pimAnd(srcObj2, stMask, srcObj2);
  status = pimAnd(srcObj3, enMask, srcObj3);
  status = pimShiftElementsLeft(srcObj3);
  status = pimAnd(srcObj2, srcObj3, srcObj2);

  std::vector<uint8_t> resultVec(vertexVector.size());

  status = pimCopyDeviceToHost(srcObj2, (void *)resultVec.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
  }

  for (unsigned coreId = 0; coreId < deviceProps.numPIMCores; ++coreId) {
    uint64_t base = (uint64_t)coreId * elementsPerRow;
    if (resultVec[base]) {
      std::cout << "vertex belongs to core " << coreId << "\n";
      break;
    }
  }
  pimFree(srcObj1);
  pimFree(srcObj2);
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Running BFS on PIM: " << params.vectorLength << "\n\n";
  std::vector<int> src1(params.vectorLength, -1), src2(params.vectorLength, 2), dst;
  if (params.shouldVerify) {
    if (params.inputFile == nullptr)
    {
      getVector(params.vectorLength, src1);
      getVector(params.vectorLength, src2);
    } else {
      std::cout << "Reading from input file is not implemented yet." << std::endl;
      return 1;
    }
  }
  if (!createDevice(params.configFile)) return 1;
  //TODO: Check if vector can fit in one iteration. Otherwise need to run addition in multiple iteration.
  vectorAddition(params.vectorLength, src1, src2, dst);
  if (params.shouldVerify) {
    // verify result
    #pragma omp parallel for
    for (unsigned i = 0; i < params.vectorLength; ++i)
    {
      int sum = src1[i] + src2[i];
      if (dst[i] != sum)
      {
        std::cout << "Wrong answer for addition: " << src1[i] << " + " << src2[i] << " = " << dst[i] << " (expected " << sum << ")" << std::endl;
      }
    }
  }

  pimShowStats();

  return 0;
}
