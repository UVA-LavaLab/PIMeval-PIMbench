// Test: C++ version of max pool in batches. 
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <getopt.h>
#include <cmath>
#include "util.h"

using namespace std;

std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();


// Params ---------------------------------------------------------------------
typedef struct Params
{
  int row, batchSize;
  char *dramConfigFile;
  char *imageMatrixFile;
  bool shouldVerify;
  bool moreDebugPrints;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./softmax-batch.out [options]"
          "\n"
          "\n    -r    vector length (default=1024)"
          "\n    -c    dram config file"
          "\n    -b    batch size (default=2)"          
          "\n    -v    should verify result with CPU"
          "\n    -f    input file containing kernel matrices (default=generates matrix with random numbers)"
          "\n    -i    input image file containing matrices (default=generates matrix with random numbers)"
	        "\n    -m    enable more debug prints (default = false)"          
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.row = 1024;
  p.batchSize = 64;
  p.dramConfigFile = nullptr;
  p.imageMatrixFile = nullptr;
  p.shouldVerify = false;
  p.moreDebugPrints = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:r:c:b:v:i:m:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'r':
      p.row = atoi(optarg);
      break;
    case 'b':
      p.batchSize = atoi(optarg);      
      break;
    case 'c':
      p.dramConfigFile = optarg;
      break;
    case 'i':
      p.imageMatrixFile = optarg;
      break;
    case 'v':
      p.shouldVerify = (*optarg == 't') ? true : false;
      break;
    case 'm':
      p.moreDebugPrints = (*optarg == 't') ? true : false;
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

void softMaxPIM(const std::vector<int> src, std::vector<int> &dst)
{
  uint64_t vectorLength = src.size();
  PimObjId srcObj = pimAlloc(PIM_ALLOC_AUTO, vectorLength, PIM_INT32);
  if (srcObj == -1)
  {
    std::cout << "Function: " << __func__ << "Abort: pimAlloc failed for obj" << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice((void *) src.data(), srcObj);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimCopyHostToDevice failed for src vector" << std::endl;
    return;
  }

  int32_t max = std::numeric_limits<int32_t>::lowest();
  status = pimRedMax(srcObj, &max);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimRedMax failed" << std::endl;
    return;
  }

  status = pimSubScalar(srcObj, srcObj, max);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimSubScalar failed" << std::endl;
    return;
  }
  
  dst.resize(vectorLength);
  status = pimCopyDeviceToHost(srcObj, (void *) dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimCopyHostToDevice failed for dst vector" << std::endl;
    return;
  }

  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for
  for (size_t i = 0; i < vectorLength; ++i)
  {
    dst[i] = std::exp(static_cast<double>(dst[i]));
  }
  auto end = std::chrono::high_resolution_clock::now();
  hostElapsedTime += (end - start);

  status = pimCopyHostToDevice((void *) dst.data(), srcObj);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimCopyHostToDevice failed for dst vector" << std::endl;
    return;
  }
  
  int32_t redsum = 0;
  status = pimRedSum(srcObj, &redsum);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimRedSum failed" << std::endl;
    return;
  }

  status = pimDivScalar(srcObj, srcObj, redsum);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimDicScalar failed" << std::endl;
    return;
  }

  status = pimCopyDeviceToHost(srcObj, (void *) dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimCopyHostToDevice failed for dst vector" << std::endl;
    return;
  }
  
  pimFree(srcObj);
}

void softmaxOnHost(const std::vector<int> &input, std::vector<int> &output)
{
    // Find the maximum value in the input vector for numerical stability
    int max_input = *std::max_element(input.begin(), input.end());

    // Compute the exponentials of each element (subtracting the max value for stability)
    std::vector<double> exponentials(input.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i)
    {
        exponentials[i] = std::exp(static_cast<double>(input[i] - max_input));
    }

    // Compute the sum of exponentials
    double sum_exponentials = 0.0;
    
    #pragma omp parallel for reduction(+:sum_exponentials)
    for (size_t i = 0; i < exponentials.size(); ++i)
    {
        sum_exponentials += exponentials[i];
    }

    // Compute softmax values and compare with PIM output
    bool mismatch_found = false;
    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i)
    {
        double softmax_val = exponentials[i] / sum_exponentials;

        // Scale softmax to integer range — assume PIM used 0–255
        int softmax_int = static_cast<int>(std::round(softmax_val * 255));

        if (softmax_int != output[i])
        {
            #pragma omp critical
            {
                std::cout << "Mismatch at index " << i
                          << ": expected " << softmax_int
                          << ", got " << output[i] << "\n";
                mismatch_found = true;
            }
        }
    }

    if (!mismatch_found)
        std::cout << "All values match.\n";

}


int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);

  std::vector<int> inputMatrix(params.row * params.batchSize);

  if (params.imageMatrixFile == nullptr)
  {
    getVector(params.row * params.batchSize, inputMatrix);
  } else {
      std::cout << "Reading from input file is not implemented yet." << std::endl;
      return 1;
  }

  if (!createDevice(params.dramConfigFile))
    return 1;
  
  std::vector<int> outVector(params.row * params.batchSize);
  softMaxPIM(inputMatrix, outVector);

  if (params.shouldVerify) {
    softmaxOnHost(inputMatrix, outVector);
  }
    
  pimShowStats();
  std::cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << std::endl;

  return 0;
}
