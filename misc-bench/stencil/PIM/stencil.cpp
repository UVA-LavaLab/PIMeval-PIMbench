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
  uint64_t stencilWidth;
  uint64_t stencilHeight;
  uint64_t numLeft;
  uint64_t numAbove;
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
          "\n    -w    horizontal stencil size (default=3)"
          "\n    -d    vertical stencil size (default=3)"
          "\n    -l    number of elements to the left of the output element for the stencil pattern, must be less than the horizontal stencil size (default=1)"
          "\n    -a    number of elements above the output element for the stencil pattern, must be less than the vertical stencil size (default=1)"
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
  p.stencilWidth = 3;
  p.stencilHeight = 3;
  p.numLeft = 1;
  p.numAbove = 1;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:n:x:y:w:d:l:a:c:i:v:")) >= 0)
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
    case 'w':
      p.stencilWidth = strtoull(optarg, NULL, 0);
      break;
    case 'd':
      p.stencilHeight = strtoull(optarg, NULL, 0);
      break;
    case 'l':
      p.numLeft = strtoull(optarg, NULL, 0);
      break;
    case 'a':
      p.numAbove = strtoull(optarg, NULL, 0);
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

// //! @brief  Shifts the elements of the input row so that necessary elements are vertically aligned
// //! @param[in]  src  The vector to shift
// //! @param[in]  stencilWidth  The horizontal width of the stencil
// //! @param[in]  numLeft  The number of elements to the left of the output element in the stencil pattern
// //! @param[in]  toAssociate  A PIM Object to associate the added data with
// //! @return  The shifted row as a list of PIM objects
// std::vector<PimObjId> createShiftedStencilRows(const std::vector<float> &src, const uint64_t stencilWidth,
//                                                const uint64_t numLeft, const PimObjId toAssociate) {
//   PimStatus status;

//   std::vector<PimObjId> result(stencilWidth);

//   for(uint64_t i=0; i<result.size(); ++i) {
//     result[i] = pimAllocAssociated(toAssociate, PIM_FP32);
//     assert(result[i] != -1);
//   }

//   status = pimCopyHostToDevice((void *)src.data(), result[numLeft]);
//   assert (status == PIM_OK);

//   for(uint64_t i=numLeft; i>0; --i) {
//     status = pimCopyObjectToObject(result[i], result[i-1]);
//     assert (status == PIM_OK);

//     status = pimShiftElementsRight(result[i-1]);
//     assert (status == PIM_OK);
//   }

//   for(uint64_t i=numLeft+1; i<result.size(); ++i) {
//     status = pimCopyObjectToObject(result[i-1], result[i]);
//     assert (status == PIM_OK);

//     status = pimShiftElementsLeft(result[i]);
//     assert (status == PIM_OK);
//   }

//   return result;
// }

//! @brief  Sums the elements to the left and right within a vector according to the horizontal stencil width
//! @param[in]  src  The vector to sum
//! @param[in]  stencilWidth  The horizontal width of the stencil
//! @param[in]  numLeft  The number of elements to the left of the output element in the stencil pattern
//! @param[in]  toAssociate  A PIM Object to associate the added data with
//! @return  The sumed PIM row
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

  // status = pimShiftElementsLeft(shiftBackup);
  // assert (status == PIM_OK);

  // status = pimShiftElementsLeft(shiftBackup);
  // assert (status == PIM_OK);

  // status = pimAdd(pimRowSum, shiftBackup, pimRowSum);
  // assert (status == PIM_OK);

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

void print_pim(PimObjId obj, uint64_t len) {
  std::vector<float> vec(len);
  PimStatus status = pimCopyDeviceToHost(obj, (void*) vec.data());
  assert (status == PIM_OK);

  for(float f : vec) {
    std::cout << f << ", ";
  }
  std::cout << std::endl;
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
  // std::cout << "radius " << radius;
  for(uint64_t i=2; i<2*radius; ++i) {
    sumStencilRow(workingPimMemory[i], rowsInSumCircularQueue[circularQueueTop], tmpPim, radius);
    status = pimAdd(runningSum, rowsInSumCircularQueue[i], runningSum);
    assert (status == PIM_OK);
    ++circularQueueTop;
  }

  uint64_t nextRowToAdd = 2*radius;

  for(uint64_t row=radius; row<workingPimMemory.size()-radius; ++row) {
    // rowsInSum.push_back(sumStencilRow(srcHost[nextRowToAdd], stencilWidth, numLeft, resultPim));
    sumStencilRow(workingPimMemory[nextRowToAdd], rowsInSumCircularQueue[circularQueueTop], tmpPim, radius);
    status = pimAdd(runningSum, rowsInSumCircularQueue[circularQueueTop], runningSum);
    // std::cout << "row: " << row << ", added from queue: ";
    // print_pim(rowsInSumCircularQueue[circularQueueTop], 10);
    // std::cout << "running sum: ";
    // print_pim(runningSum, 10);
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
//! @param[in]  dstHost  The resultant stencil grid
//! @param[in]  stencilPattern  The stencil pattern to apply
//! @param[in]  numLeft  The number of elements to the left of the output element in the stencil pattern
//! @param[in]  numAbove  The number of elements above the output element in the stencil pattern
void stencil(const std::vector<std::vector<float>> &srcHost, std::vector<std::vector<float>> &dstHost, const uint64_t numRows,
              const uint64_t iterations, const uint64_t radius) {
  PimStatus status;
  
  assert(!srcHost.empty());
  assert(!srcHost[0].empty());
  assert(srcHost.size() == dstHost.size());
  assert(srcHost[0].size() == dstHost[0].size());
  // assert(srcHost.size() <= 20);

  const uint64_t gridHeight = srcHost.size();
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

  // uint64_t workingPimMemoryIdx = 0;
  // for(uint64_t srcHostRow = 0; srcHostRow < srcHost.size(); ++srcHostRow) {
  //   status = pimCopyHostToDevice((void*) srcHost[srcHostRow].data(), workingPimMemory[workingPimMemoryIdx]);
  //   assert (status == PIM_OK);
  //   ++workingPimMemoryIdx;
  // }
  // computeStencilChunkIteration(workingPimMemory, rowsInSumCircularQueue, tmpPim, runningSum, stencilAreaToMultiplyPim, radius);
  // workingPimMemoryIdx = 1;
  // for(uint64_t srcHostRow = 1; srcHostRow < srcHost.size()-1; ++srcHostRow) {
  //   status = pimCopyDeviceToHost(workingPimMemory[workingPimMemoryIdx], (void*) dstHost[srcHostRow].data());
  //   assert (status == PIM_OK);
  //   ++workingPimMemoryIdx;
  // }

  // Should (untested) compute stenil for currIterations for the whole grid. Will be run in a loop to compute all the iterations, as only (max iterations per loop) iterations can be run for each loop
  const uint64_t currIterations = 2;
  const uint64_t invalidResultsTop = currIterations - 1 + radius;
  const uint64_t maxUsableResults = workingPimMemory.size() - 2*invalidResultsTop;
  uint64_t firstRowSrc = 0;
  for(;;) {
    // copy srcHost [firstRowUsable - invalidResultsTop, min(firstRowUsable - invalidResultsTop+wpmSz, end of rows)) into first slots of working memory
      // rowsThisIter = numcopied
    std::cout << "first row src: " << firstRowSrc << ", srcHost.size(): " << srcHost.size() << std::endl;
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
    // computeStencilChunkIteration x currIters
    for(uint64_t iterNum = 0; iterNum < currIterations; ++iterNum) {
      computeStencilChunkIteration(workingPimMemory, rowsInSumCircularQueue, tmpPim, runningSum, stencilAreaToMultiplyPim, radius);
    }
    // copy range wpm [invalidResultsTop, (used wpm size)-invalidResultsTop) into dstHost [firstRowUsable, firstRowUsable+rowsThisIter)
    workingPimMemoryIdx = invalidResultsTop;
    for(uint64_t srcHostRow = firstRowUsableSrc; srcHostRow < firstRowUsableSrc + usableRowsThisIter; ++srcHostRow) {
      status = pimCopyDeviceToHost(workingPimMemory[workingPimMemoryIdx], (void*) dstHost[srcHostRow].data());
      assert (status == PIM_OK);
      ++workingPimMemoryIdx;
    }
    // firstRowUsable += rowsOnIter
    firstRowSrc += usableRowsThisIter;
  }

  

  // PimObjId resultPim = pimAlloc(PIM_ALLOC_AUTO, gridWidth, PIM_FP32);
  // assert(resultPim != -1);

  // // Handle special case
  // if(stencilHeight == 1) {
  //   for(size_t i=0; i<gridHeight; ++i) {
  //     PimObjId summedRow = sumStencilRow(srcHost[i], stencilWidth, numLeft, resultPim);

  //     status = pimMulScalar(summedRow, resultPim, stencilAreaToMultiply);
  //     assert (status == PIM_OK);

  //     status = pimCopyDeviceToHost(resultPim, dstHost[i].data());
  //     assert (status == PIM_OK);

  //     pimFree(summedRow);
  //   }

  //   pimFree(resultPim);
  //   return;
  // }

  // PimObjId runningSum = pimAllocAssociated(resultPim, PIM_FP32);
  // assert(runningSum != -1);

  // std::list<PimObjId> rowsInSum;
  // rowsInSum.push_back(sumStencilRow(srcHost[0], stencilWidth, numLeft, resultPim));
  // rowsInSum.push_back(sumStencilRow(srcHost[1], stencilWidth, numLeft, resultPim));
  // status = pimAdd(rowsInSum.front(), rowsInSum.back(), runningSum);
  // assert (status == PIM_OK);

  // for(uint64_t i=2; i<stencilHeight-1; ++i) {
  //   rowsInSum.push_back(sumStencilRow(srcHost[i], stencilWidth, numLeft, resultPim));
  //   status = pimAdd(runningSum, rowsInSum.back(), runningSum);
  //   assert (status == PIM_OK);
  // }

  // uint64_t nextRowToAdd = stencilHeight-1;

  // for(uint64_t row=numAbove; row<gridHeight-numBelow; ++row) {
  //   rowsInSum.push_back(sumStencilRow(srcHost[nextRowToAdd], stencilWidth, numLeft, resultPim));
  //   status = pimAdd(runningSum, rowsInSum.back(), runningSum);
  //   assert (status == PIM_OK);
  //   ++nextRowToAdd;

  //   status = pimMulScalar(runningSum, resultPim, stencilAreaToMultiply);
  //   assert (status == PIM_OK);

  //   status = pimCopyDeviceToHost(resultPim, (void *) dstHost[row].data());
  //   assert (status == PIM_OK);
    
  //   if(row+1<gridHeight-numBelow) {
  //     status = pimSub(runningSum, rowsInSum.front(), runningSum);
  //     assert (status == PIM_OK);
  //     status = pimFree(rowsInSum.front());
  //     assert (status == PIM_OK);
  //     rowsInSum.pop_front();
  //   }
  // }

  // while(!rowsInSum.empty()) {
  //   status = pimFree(rowsInSum.front());
  //   assert (status == PIM_OK);
  //   rowsInSum.pop_front();
  // }
}

void stencilCpu(std::vector<std::vector<float>>& src, std::vector<std::vector<float>>& dst, const uint64_t iterations, const uint64_t radius) {
  // Only compute when stencil is fully in range
  const uint64_t startY = radius;
  const uint64_t endY = src.size() - radius;
  const uint64_t startX = radius;
  const uint64_t endX = src[0].size() - radius;
  const uint64_t stencilAreaInt = (2 * radius + 1) * (2 * radius + 1);
  const float stencilAreaInverseFloat = 1.0f / static_cast<float>(stencilAreaInt);

  for(uint64_t iter=0; iter<iterations; ++iter) {
    #pragma omp parallel for collapse(2)
    for(uint64_t gridY=startY; gridY<endY; ++gridY) {
      for(uint64_t gridX=startX; gridX<endX; ++gridX) {
        float resCPU = 0.0f;
        for(uint64_t stencilY=gridY-1; stencilY<=gridY+1; ++stencilY) {
          for(uint64_t stencilX=gridX-1; stencilX<=gridX+1; ++stencilX) {
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
  std::cout << "Stencil Size: " << params.stencilHeight << "x" << params.stencilWidth << std::endl;
  std::cout << "Num Above: " << params.numAbove << ", Num Left: " << params.numLeft << std::endl;

  std::vector<std::vector<float>> x, y;

  if (params.inputFile == nullptr)
  {
    // Fill in random grid
    x.resize(params.gridHeight);
    for(size_t i=0; i<x.size(); ++i) {
      x[i].resize(params.gridWidth);
    }

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

  y.resize(x.size());
  for(size_t i=0; i<y.size(); ++i) {
    y[i].resize(x[0].size());
  }

  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  assert(status == PIM_OK);
  // void stencil(const std::vector<std::vector<float>> &srcHost, std::vector<std::vector<float>> &dstHost, const uint64_t numRows,
  // const uint64_t iterations, const uint64_t radius)
  stencil(x, y, 2 * deviceProp.numRowPerSubarray, params.iterations, 1);

  if (params.shouldVerify) 
  {
    std::vector<std::vector<float>> cpuY;
    cpuY.resize(y.size(), std::vector<float>(y[0].size(), 0));
    stencilCpu(x, cpuY, params.iterations, 1);

    bool ok = true;

    // Only compute when stencil is fully in range
    const uint64_t startY = 1 + params.iterations - 1;
    const uint64_t endY = params.gridHeight - (1 + params.iterations - 1);
    const uint64_t startX = 1 + params.iterations - 1;
    const uint64_t endX = params.gridWidth - (1 + params.iterations - 1);

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