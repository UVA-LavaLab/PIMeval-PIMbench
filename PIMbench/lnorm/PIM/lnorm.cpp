// Test: C++ version of matrix vector multiplication
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "util.h"
#include "libpimeval.h"
#include <chrono>
#include <cmath>

std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();
//auto start_cpu, stop_cpu;

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
          "\nUsage:  ./lnorm.out [options]"
          "\n"
          "\n    -l    vectorLength (default=128 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing two vectors (default=generates vector with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.vectorLength = 128;
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

// Newton-Raphson iterative integer square root
uint32_t newton_sqrt(uint32_t x) {
  if (x == 0) return 0;  // Handle zero case

  uint32_t guess = x; // Initial guess
  uint32_t prev_guess = 0;

  while (guess != prev_guess) { // Continue until convergence
      prev_guess = guess;
      guess = (guess + x / guess) / 2; // Newton-Raphson iteration
  }

  //std::cout << "newton sqrt: " << guess << std::endl;
  return guess;
}

void lnorm(uint64_t vectorLength, std::vector<int> &srcVector, std::vector<int> &dst)
{
  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, vectorLength, PIM_INT32);
  if (srcObj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId tempObj1 = pimAllocAssociated(srcObj1, PIM_INT32);
  if (tempObj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId dstObj = pimAllocAssociated(srcObj1, PIM_INT32);
  if (dstObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimStatus status;

  status = pimCopyHostToDevice((void *)srcVector.data(), srcObj1);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  //mean
  int32_t sum = 0;
  status = pimRedSum(srcObj1, static_cast<void*>(&sum), 0, vectorLength);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  } 
  auto start_cpu = std::chrono::high_resolution_clock::now();
  int32_t mean = sum/vectorLength;
  std::cout << "mean " << mean << " sum " << sum <<std::endl;
  auto stop_cpu = std::chrono::high_resolution_clock::now();
  hostElapsedTime += (stop_cpu - start_cpu);
  
  status = pimSubScalar(srcObj1, tempObj1, mean);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  } 
  
  status = pimMul(tempObj1, tempObj1, dstObj);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  } 

  int32_t sum2 = 0;
  status = pimRedSum(dstObj, static_cast<void*>(&sum2), 0, vectorLength);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  
  start_cpu = std::chrono::high_resolution_clock::now(); 

  int32_t variance = sum2/vectorLength;
  int32_t sqrt_var = newton_sqrt(variance + 1);
  std::cout << "sqrt_var " << sqrt_var << " var " << variance <<std::endl;

  stop_cpu = std::chrono::high_resolution_clock::now();
  hostElapsedTime += (stop_cpu - start_cpu);

  // Scale sqrt_variance
  status = pimDivScalar(tempObj1, dstObj, sqrt_var);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  } 


  dst.resize(vectorLength);
  status = pimCopyDeviceToHost(dstObj, (void *)dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
  }
  pimFree(srcObj1);
  pimFree(tempObj1);
  pimFree(dstObj);
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Running LNORM for vector of size: " << params.vectorLength << std::endl;

  std::vector<int> srcVector (params.vectorLength, 1), resultVector;

  if (params.shouldVerify) {
    if (params.inputFile == nullptr)
    {
      getVector(params.vectorLength, srcVector);
    }
    else
    {
      std::cout << "Reading from input file is not implemented yet." << std::endl;
      return 1;
    }
  }


  if (!createDevice(params.configFile))
  {
    return 1;
  }

  // TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iteration.
  lnorm(params.vectorLength, srcVector, resultVector);

  if (params.shouldVerify)
  {
    bool shouldBreak = false; // shared flag variable

    // verify result

      std::vector<int> result (params.vectorLength, 0);
      std::vector<int> src_minus_mean (params.vectorLength, 0);
      std::vector<int> sq_src_minus_mean (params.vectorLength, 0);
      
      int32_t sum = 0;

      for (size_t i = 0; i < params.vectorLength; i++) {
          sum += srcVector[i];
      } 

      int32_t mean = sum / params.vectorLength; 

      for (size_t i = 0; i < params.vectorLength; i++) {
        src_minus_mean[i] = srcVector[i] - mean;  
      }

      for (size_t i = 0; i < params.vectorLength; i++) {
        sq_src_minus_mean[i] = (int32_t)(src_minus_mean[i]*src_minus_mean[i]);  
      }

      int32_t sum2 = 0;
      for (size_t i = 0; i < params.vectorLength; i++) {
        sum2 += sq_src_minus_mean[i];
      } 

      int32_t var = sum2/params.vectorLength;

      int32_t sqrt_var = newton_sqrt(var+1);
      if(sqrt_var==0){
        sqrt_var = 1;
      }

      // layer norm
      for (size_t i = 0; i < params.vectorLength; i++) {
          result[i] = src_minus_mean[i] / (sqrt_var);  // Prevent division by zero
      }

    for (size_t i = 0; i < params.vectorLength; i++)
    {
      if (result[i] != resultVector[i])
      {
        #pragma omp critical
        {
          if (!shouldBreak)
          { // check the flag again in a critical section
            std::cout << "Wrong answer: " << resultVector[i] << " (expected " << result[i] << ")" << std::endl;
            shouldBreak = true; // set the flag to true
          }
        }
      }
    }
    

    if (!shouldBreak) {
      std::cout << "\n\nCorrect Answer!!\n\n";
    }
  }

  pimShowStats();
  std::cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << endl;

  return 0;
}
