// Test: C++ version of the stencil
// Copyright (c) 2025 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#include <cassert>
#include <type_traits>
#include <queue>
#include <random>
#include <limits>
#include <algorithm>
#include <list>
#include <cstring>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "util.h"
#include "libpimeval.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t iterations;
  uint64_t gridWidth;
  uint64_t gridHeight;
  uint64_t radius;
  const char *configFile;
  const char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./stencil.out [options]"
          "\n"
          "\n    -n    iterations (default=10 iterations)"
          "\n    -x    grid width (default=2048 elements)"
          "\n    -y    grid height (default=2048 elements)"
          "\n    -r    stencil radius (default=1)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing a 2d array (default=random)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.iterations = 10;
  p.gridWidth = 2048;
  p.gridHeight = 2048;
  p.radius = 1;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = true;

  int opt;
  while ((opt = getopt(argc, argv, "h:n:x:y:r:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'n':
      p.iterations = strtoull(optarg, NULL, 0);
      break;
    case 'x':
      p.gridWidth = strtoull(optarg, NULL, 0);
      break;
    case 'y':
      p.gridHeight = strtoull(optarg, NULL, 0);
      break;
    case 'r':
      p.radius= strtoull(optarg, NULL, 0);
      break;
    case 'c':
      p.configFile = optarg;
      break;
    case 'i':
      p.inputFile = optarg;
      break;
    case 'v':
      p.shouldVerify = (*optarg == 't');
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

//! @brief  Sums the neighbors of each element in a stencil row to compute the horizontal stencil sum 
//! @param[in]  mid  PIM row to be summed
//! @param[out]  pimRowSum  The resultant PIM object to place the sum into
//! @param[in,out]  shiftBackup  Temporary PIM object used for calculations
//! @param[in]  radius  The stencil radius
void sumStencilRow(PimObjId mid, PimObjId pimRowSum, PimObjId shiftBackup, const uint64_t radius) {
  PimStatus status;

  if(radius == 0) {
    return;
  }
  
  status = pimCopyObjectToObject(mid, shiftBackup);
  assert (status == PIM_OK);

  status = pimShiftElementsRight(shiftBackup);
  assert (status == PIM_OK);

  status = pimAdd(mid, shiftBackup, pimRowSum);
  assert (status == PIM_OK);

  for(uint64_t shiftIter=1; shiftIter<radius; ++shiftIter) {
    status = pimShiftElementsRight(shiftBackup);
    assert (status == PIM_OK);

    status = pimAdd(pimRowSum, shiftBackup, pimRowSum);
    assert (status == PIM_OK);
  }

  status = pimCopyObjectToObject(mid, shiftBackup);
  assert (status == PIM_OK);

  for(uint64_t shiftIter=0; shiftIter<radius; ++shiftIter) {
    status = pimShiftElementsLeft(shiftBackup);
    assert (status == PIM_OK);

    status = pimAdd(pimRowSum, shiftBackup, pimRowSum);
    assert (status == PIM_OK);
  }
}

void computeStencilChunkIteration(std::vector<PimObjId>& workingPimMemory, std::vector<PimObjId>& rowsInSumCircularQueue, PimObjId tmpPim, PimObjId runningSum, const uint64_t stencilAreaToMultiplyPim, const uint64_t radius) {
  PimStatus status;
  uint64_t circularQueueTop = 0;
  uint64_t circularQueueBot = 0;

  sumStencilRow(workingPimMemory[0], rowsInSumCircularQueue[circularQueueTop], tmpPim, radius);
  ++circularQueueTop;
  sumStencilRow(workingPimMemory[1], rowsInSumCircularQueue[circularQueueTop], tmpPim, radius);
  ++circularQueueTop;
  status = pimAdd(rowsInSumCircularQueue[0], rowsInSumCircularQueue[1], runningSum);
  assert (status == PIM_OK);

  for(uint64_t i=2; i<2*radius; ++i) {
    sumStencilRow(workingPimMemory[i], rowsInSumCircularQueue[circularQueueTop], tmpPim, radius);
    status = pimAdd(runningSum, rowsInSumCircularQueue[circularQueueTop], runningSum);
    assert (status == PIM_OK);
    ++circularQueueTop;
  }

  uint64_t nextRowToAdd = 2*radius;

  for(uint64_t row=radius; row<workingPimMemory.size()-radius; ++row) {
    sumStencilRow(workingPimMemory[nextRowToAdd], rowsInSumCircularQueue[circularQueueTop], tmpPim, radius);

    status = pimAdd(runningSum, rowsInSumCircularQueue[circularQueueTop], runningSum);
    assert (status == PIM_OK);

    circularQueueTop = (1+circularQueueTop) % rowsInSumCircularQueue.size();
    ++nextRowToAdd;

    status = pimMulScalar(runningSum, workingPimMemory[row], stencilAreaToMultiplyPim);
    assert (status == PIM_OK);
    
    if(row+1<workingPimMemory.size()-radius) {
      status = pimSub(runningSum, rowsInSumCircularQueue[circularQueueBot], runningSum);
      assert (status == PIM_OK);
      circularQueueBot = (1+circularQueueBot) % rowsInSumCircularQueue.size();
    }
  }
}

//! @brief  Computes a stencil pattern over a 2d array
//! @param[in]  srcHost  The input stencil grid
//! @param[out]  dstHost  The resultant stencil grid
//! @param[in]  numRows  Number of PIM rows that objects can be associated within
//! @param[in]  iterations  Number of iterations to run the stencil pattern for
//! @param[in]  radius  The radius of the stencil pattern
void stencil(const std::vector<std::vector<float>> &srcHost, std::vector<std::vector<float>> &dstHost,
              const uint64_t numRows, const uint64_t iterations, const uint64_t radius) {
  PimStatus status;
  
  assert(!srcHost.empty());
  assert(!srcHost[0].empty());
  assert(srcHost.size() == dstHost.size());
  assert(srcHost[0].size() == dstHost[0].size());

  const uint64_t gridWidth = srcHost[0].size();

  const uint64_t stencilAreaInt = (2 * radius + 1) * (2 * radius + 1);
  const float stencilAreaFloat = 1.0f / static_cast<float>(stencilAreaInt);
  uint32_t tmp;
  std::memcpy(&tmp, &stencilAreaFloat, sizeof(float));
  const uint64_t stencilAreaToMultiplyPim = static_cast<uint64_t>(tmp);
  constexpr uint64_t numIterationsPerPim = 5;

  PimObjId tmpPim = pimAlloc(PIM_ALLOC_AUTO, gridWidth, PIM_FP32);
  assert(tmpPim != -1);
  PimObjId runningSum = pimAllocAssociated(tmpPim, PIM_FP32);
  assert(runningSum != -1);

  std::vector<PimObjId> rowsInSumCircularQueue(2*radius+1);
  for(uint64_t i=0; i<rowsInSumCircularQueue.size(); ++i) {
    rowsInSumCircularQueue[i] = pimAllocAssociated(tmpPim, PIM_FP32);
    assert(rowsInSumCircularQueue[i] != -1);
  }

  std::vector<PimObjId> workingPimMemory(20); // TODO: Set to a better number, num associable - num used other
  for(uint64_t i=0; i<workingPimMemory.size(); ++i) {
    workingPimMemory[i] = pimAllocAssociated(tmpPim, PIM_FP32);
    assert(workingPimMemory[i] != -1);
  }

  // Should (untested) compute stenil for currIterations for the whole grid. Will be run in a loop to compute all the iterations, as only (max iterations per loop) iterations can be run for each loop
  const uint64_t currIterations = iterations;
  const uint64_t invalidResultsTop = radius * currIterations;

  uint64_t firstRowSrc = 0;
  for(;;) {
    const uint64_t firstRowUsableSrc = firstRowSrc + invalidResultsTop;
    if(firstRowUsableSrc + invalidResultsTop >= srcHost.size()) {
      break;
    }
    const uint64_t totalRowsThisIter = min(srcHost.size(), firstRowSrc + workingPimMemory.size()) - firstRowSrc;
    const uint64_t usableRowsThisIter = totalRowsThisIter - 2*invalidResultsTop;
    uint64_t workingPimMemoryIdx = 0;
    for(uint64_t srcHostRow = firstRowSrc; srcHostRow < firstRowSrc + totalRowsThisIter; ++srcHostRow) {
      status = pimCopyHostToDevice((void*) srcHost[srcHostRow].data(), workingPimMemory[workingPimMemoryIdx]);
      assert (status == PIM_OK);
      ++workingPimMemoryIdx;
    }

    for(uint64_t iterNum = 0; iterNum < currIterations; ++iterNum) {
      computeStencilChunkIteration(workingPimMemory, rowsInSumCircularQueue, tmpPim, runningSum, stencilAreaToMultiplyPim, radius);
    }

    workingPimMemoryIdx = invalidResultsTop;
    for(uint64_t srcHostRow = firstRowUsableSrc; srcHostRow < firstRowUsableSrc + usableRowsThisIter; ++srcHostRow) {
      status = pimCopyDeviceToHost(workingPimMemory[workingPimMemoryIdx], (void*) dstHost[srcHostRow].data());
      assert (status == PIM_OK);
      ++workingPimMemoryIdx;
    }

    firstRowSrc += usableRowsThisIter;
  }
}

void stencilCpu(std::vector<std::vector<float>>& src, std::vector<std::vector<float>>& dst, const uint64_t iterations, const uint64_t radius) {
  const uint64_t stencilAreaInt = (2 * radius + 1) * (2 * radius + 1);
  const float stencilAreaInverseFloat = 1.0f / static_cast<float>(stencilAreaInt);

  for(uint64_t iter=1; iter<=iterations; ++iter) {
    // Only compute when stencil is fully in range
    const uint64_t startY = radius*iter;
    const uint64_t endY = src.size() - startY;
    const uint64_t startX = radius*iter;
    const uint64_t endX = src[0].size() - startX;
    #pragma omp parallel for collapse(2)
    for(uint64_t gridY=startY; gridY<endY; ++gridY) {
      for(uint64_t gridX=startX; gridX<endX; ++gridX) {
        float resCPU = 0.0f;
        for(uint64_t stencilY=gridY-radius; stencilY<=gridY+radius; ++stencilY) {
          for(uint64_t stencilX=gridX-radius; stencilX<=gridX+radius; ++stencilX) {
            resCPU += src[stencilY][stencilX];
          }
        }
        dst[gridY][gridX] = resCPU * stencilAreaInverseFloat;
      }
    }
    std::swap(src, dst);
  }
  std::swap(src, dst);
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);

  std::cout << "Running PIM stencil for grid: " << params.gridHeight << "x" << params.gridWidth << std::endl;
  std::cout << "Stencil Radius: " << params.radius << std::endl;

  std::vector<std::vector<float>> x, y;

  if (params.inputFile == nullptr)
  {
    // Fill in random grid
    x.resize(params.gridHeight, std::vector<float>(params.gridWidth));

    #pragma omp parallel
    {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dist(0.0f, 10000.0f);
      
      #pragma omp for
      for(size_t i=0; i<params.gridHeight; ++i) {
        for(size_t j=0; j<params.gridWidth; ++j) {
          x[i][j] = dist(gen);
        }
      }
    }
  } 
  else 
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 1;
  }
  
  if (!createDevice(params.configFile))
  {
    return 1;
  }

  y.resize(x.size(), std::vector<float>(x[0].size()));

  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  assert(status == PIM_OK);

  stencil(x, y, 2 * deviceProp.numRowPerSubarray, params.iterations, params.radius);

  if (params.shouldVerify) 
  {
    std::vector<std::vector<float>> cpuY(y.size(), std::vector<float>(y[0].size()));
    stencilCpu(x, cpuY, params.iterations, params.radius);

    bool ok = true;

    // Only compute when stencil is fully in range
    const uint64_t startY = params.radius * params.iterations;
    const uint64_t endY = params.gridHeight - startY;
    const uint64_t startX = params.radius * params.iterations;
    const uint64_t endX = params.gridWidth - startX;

    std::cout << std::fixed << std::setprecision(10);

    #pragma omp parallel for collapse(2)
    for(uint64_t gridY=startY; gridY<endY; ++gridY) {
      for(uint64_t gridX=startX; gridX<endX; ++gridX) {
        constexpr float acceptableDelta = 0.1f;
        if (std::abs(cpuY[gridY][gridX] - y[gridY][gridX]) > acceptableDelta)
        {
          #pragma omp critical
          {
            std::cout << "Wrong answer: " << y[gridY][gridX] << " (expected " << cpuY[gridY][gridX] << ") at position (" << gridX << ", " << gridY << ")" << std::endl;
            ok = false;
          }
        }
      }
    }
    if(ok) {
      std::cout << "Correct for stencil!" << std::endl;
    }
  }

  pimShowStats();

  return 0;
}