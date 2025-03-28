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
  while ((opt = getopt(argc, argv, "h:x:y:w:d:l:a:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
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

//! @brief  Shifts the elements of the input row so that necessary elements are vertically aligned
//! @param[in]  src  The vector to shift
//! @param[in]  stencilWidth  The horizontal width of the stencil
//! @param[in]  numLeft  The number of elements to the left of the output element in the stencil pattern
//! @param[in]  toAssociate  A PIM Object to associate the added data with
//! @return  The shifted row as a list of PIM objects
std::vector<PimObjId> createShiftedStencilRows(const std::vector<float> &src, const uint64_t stencilWidth,
                                               const uint64_t numLeft, const PimObjId toAssociate) {
  PimStatus status;

  std::vector<PimObjId> result(stencilWidth);

  for(uint64_t i=0; i<result.size(); ++i) {
    result[i] = pimAllocAssociated(toAssociate, PIM_FP32);
    assert(result[i] != -1);
  }

  status = pimCopyHostToDevice((void *)src.data(), result[numLeft]);
  assert (status == PIM_OK);

  for(uint64_t i=numLeft; i>0; --i) {
    status = pimCopyObjectToObject(result[i], result[i-1]);
    assert (status == PIM_OK);

    status = pimShiftElementsRight(result[i-1]);
    assert (status == PIM_OK);
  }

  for(uint64_t i=numLeft+1; i<result.size(); ++i) {
    status = pimCopyObjectToObject(result[i-1], result[i]);
    assert (status == PIM_OK);

    status = pimShiftElementsLeft(result[i]);
    assert (status == PIM_OK);
  }

  return result;
}

//! @brief  Computes a stencil pattern over a 2d array
//! @param[in]  srcHost  The input stencil grid
//! @param[in]  dstHost  The resultant stencil grid
//! @param[in]  stencilPattern  The stencil pattern to apply
//! @param[in]  numLeft  The number of elements to the left of the output element in the stencil pattern
//! @param[in]  numAbove  The number of elements above the output element in the stencil pattern
void stencil(const std::vector<std::vector<float>> &srcHost, std::vector<std::vector<float>> &dstHost,
             const std::vector<std::vector<float>> &stencilPattern, const uint64_t numLeft, const uint64_t numAbove) {
  PimStatus status;
  
  assert(!srcHost.empty());
  assert(!srcHost[0].empty());
  assert(srcHost.size() == dstHost.size());
  assert(srcHost[0].size() == dstHost[0].size());
  assert(!stencilPattern.empty());
  assert(!stencilPattern[0].empty());
  assert(stencilPattern.size() > numAbove);
  assert(stencilPattern[0].size() > numLeft);

  const uint64_t gridHeight = srcHost.size();
  const uint64_t gridWidth = srcHost[0].size();
  const uint64_t stencilHeight = stencilPattern.size();
  const uint64_t stencilWidth = stencilPattern[0].size();
  const uint64_t numBelow = stencilHeight - numAbove - 1;
  
  // PIM API only supports passing scalar values through uint64_t
  std::vector<std::vector<uint64_t>> stencilPatternConverted(stencilHeight);
  for(uint64_t y=0; y<stencilHeight; ++y) {
    stencilPatternConverted[y].resize(stencilWidth);
    for(uint64_t x=0; x<stencilWidth; ++x) {
      // Use memcpy as to not break aliasing rules
      uint32_t tmp;
      std::memcpy(&tmp, &stencilPattern[y][x], sizeof(float));
      stencilPatternConverted[y][x] = static_cast<uint64_t>(tmp);
    }
  }

  PimObjId resultPim = pimAlloc(PIM_ALLOC_AUTO, gridWidth, PIM_FP32);
  assert(resultPim != -1);

  PimObjId tempPim = pimAllocAssociated(resultPim, PIM_FP32);
  assert(tempPim != -1);

  std::list<std::vector<PimObjId>> shiftedRows;

  for(uint64_t i=0; i<stencilHeight-1; ++i) {
    shiftedRows.push_back(createShiftedStencilRows(srcHost[i], stencilWidth, numLeft, resultPim));
  }

  uint64_t nextRowToAdd = stencilHeight-1;

  for(uint64_t row=numAbove; row<gridHeight-numBelow; ++row) {
    shiftedRows.push_back(createShiftedStencilRows(srcHost[nextRowToAdd], stencilWidth, numLeft, resultPim));
    ++nextRowToAdd;

    uint64_t stencilY = 0;
    for(std::vector<PimObjId> &shiftedRow : shiftedRows) {
      for(uint64_t stencilX = 0; stencilX < stencilWidth; ++stencilX) {
        if(stencilY == 0 && stencilX == 0) {
          status = pimMulScalar(shiftedRow[stencilX], resultPim, stencilPatternConverted[stencilY][stencilX]);
          assert (status == PIM_OK);
        } else {
          status = pimMulScalar(shiftedRow[stencilX], tempPim, stencilPatternConverted[stencilY][stencilX]);
          assert (status == PIM_OK);

          status = pimAdd(resultPim, tempPim, resultPim);
          assert (status == PIM_OK);
        }
      }
      ++stencilY;
    }

    status = pimCopyDeviceToHost(resultPim, (void *) dstHost[row].data());
    assert (status == PIM_OK);
    
    for(PimObjId objToFree : shiftedRows.front()) {
      pimFree(objToFree);
    }
    shiftedRows.pop_front();
  }

  while(!shiftedRows.empty()) {
    for(PimObjId objToFree : shiftedRows.front()) {
      pimFree(objToFree);
    }
    shiftedRows.pop_front();
  }
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);

  std::cout << "Running PIM stencil for grid: " << params.gridHeight << "x" << params.gridWidth << std::endl;
  std::cout << "Stencil Size: " << params.stencilHeight << "x" << params.stencilWidth << std::endl;
  std::cout << "Num Above: " << params.numAbove << ", Num Left: " << params.numLeft << std::endl;

  std::vector<std::vector<float>> x, y;
  std::vector<std::vector<float>> stencilPattern;
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

    // Fill in random stencil pattern
    stencilPattern.resize(params.stencilHeight);
    for(size_t i=0; i<stencilPattern.size(); ++i) {
      stencilPattern[i].resize(params.stencilWidth);
    }

    #pragma omp parallel
    {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      
      #pragma omp for
      for(size_t i=0; i<params.stencilHeight; ++i) {
        for(size_t j=0; j<params.stencilWidth; ++j) {
          stencilPattern[i][j] = dist(gen);
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

  // TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iteration.
  y.resize(x.size());
  for(size_t i=0; i<y.size(); ++i) {
    y[i].resize(x[0].size());
  }

  stencil(x, y, stencilPattern, params.numLeft, params.numAbove);

  if (params.shouldVerify) 
  {
    bool ok = true;

    // Only compute when stencil is fully in range
    const uint64_t startY = params.numAbove;
    const uint64_t endY = params.gridHeight - (params.stencilHeight - params.numAbove - 1);
    const uint64_t startX = params.numLeft;
    const uint64_t endX = params.gridWidth - (params.stencilWidth - params.numLeft - 1);

    #pragma omp parallel for collapse(2)
    for(uint64_t gridY=startY; gridY<endY; ++gridY) {
      for(uint64_t gridX=startX; gridX<endX; ++gridX) {
        float resCPU = 0.0f;
        for(uint64_t stencilY=0; stencilY<params.stencilHeight; ++stencilY) {
          for(uint64_t stencilX=0; stencilX<params.stencilWidth; ++stencilX) {
            resCPU += stencilPattern[stencilY][stencilX] * x[gridY + stencilY - params.numAbove][gridX + stencilX - params.numLeft];
          }
        }
        if (resCPU != y[gridY][gridX])
        {
          #pragma omp critical
          {
            std::cout << "Wrong answer: " << y[gridY][gridX] << " (expected " << resCPU << ")" << std::endl;
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