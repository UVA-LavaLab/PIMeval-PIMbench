// Test: C++ version of prefix sum
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "../../util/util.h"
#include "libpimeval.h"
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#include <chrono>

std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();


#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace std;

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
          "\nUsage:  ./prefix-sum.out [options]"
          "\n"
          "\n    -l    input size (default=2048 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing interger vector (default=generates datapoints with random numbers)"
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

void prefixSum(vector<int> &even, vector<int> &odd,vector<int> &deviceoutput, uint64_t len)
{
 
  PimObjId evenObj = pimAlloc(PIM_ALLOC_AUTO, len, PIM_INT32);
  if (evenObj == -1)
  {
    std::cerr << "Abort: Failed to allocate memory on PIM." << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice((void *)even.data(), evenObj);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy data to PIM." << std::endl;
    return;
  }
  
  PimObjId oddObj = pimAllocAssociated(evenObj , PIM_INT32);
  if (oddObj == -1)
  {
    std::cerr << "Abort: Failed to allocate memory on PIM." << std::endl;
    return;
  }

  status = pimCopyHostToDevice((void *)odd.data(), oddObj);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy data to PIM." << std::endl;
  } 

  PimObjId outObj = pimAllocAssociated(evenObj, PIM_INT32);
  if (outObj == -1)
  {
    std::cerr << "Abort: Failed to allocate memory on PIM." << std::endl;
    return;
  }

  status = pimCopyHostToDevice((void *)deviceoutput.data(), outObj);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy data to PIM." << std::endl;
  } 
  
  //PIM Add
  status = pimAdd(evenObj, oddObj, outObj);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to perform PIM addition." << std::endl;
    return;
  }
  //Copy results back to Host
  status = pimCopyDeviceToHost(outObj, (void *)deviceoutput.data());
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy prefix sum result from PIM." << std::endl;
    return;
  }

  // Clean up PIM objects
  pimFree(evenObj);
  pimFree(oddObj);
  pimFree(outObj);
}


void downsweep(vector<int> &odd2, vector<int> &even2, uint64_t len)
{
 
  PimObjId evenObj = pimAlloc(PIM_ALLOC_AUTO, len, PIM_INT32);
  if (evenObj == -1)
  {
    std::cerr << "Abort: Failed to allocate memory on PIM." << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice((void *)even2.data(), evenObj);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy data to PIM." << std::endl;
    return;
  }

  PimObjId oddObj = pimAllocAssociated(evenObj, PIM_INT32);
  if (evenObj == -1)
  {
    std::cerr << "Abort: Failed to allocate memory on PIM." << std::endl;
    return;
  }

  status = pimCopyHostToDevice((void *)odd2.data(), oddObj);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy data to PIM." << std::endl;
  } 
  
  //PIM Add
  status = pimAdd(evenObj, oddObj, evenObj);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to perform PIM addition." << std::endl;
    return;
  }
 
  //Copy results back to Host
  status = pimCopyDeviceToHost(evenObj, (void *)even2.data());
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy prefix sum result from PIM." << std::endl;
    return;
  }
   
  // Clean up PIM objects
  pimFree(evenObj);
  pimFree(oddObj);
}


int main(int argc, char *argv[])
{
    struct Params params = getInputParams(argc, argv);
    std::vector<int> input;

    if (params.inputFile == nullptr) {
        getVector(params.vectorLength, input);
    } else {
        std::cout << "Reading from input file is not implemented yet." << std::endl;
        return 1;
    }

    uint64_t len = input.size();
    std::vector<int> deviceoutput;
    std::vector<int> hostoutput(len);
    std::vector<int> intermeadiate_results;
    std::vector<int> host_device_merged;

    auto start_cpu = std::chrono::high_resolution_clock::now();

    for (uint64_t i = 0; i < input.size(); i++) {
        deviceoutput.push_back(0);
    }

    hostoutput[0] = input[0];
    for (uint64_t i = 0; i < input.size(); i++) {
        hostoutput[i + 1] = hostoutput[i] + input[i + 1];
        intermeadiate_results.push_back(input[i]);
        host_device_merged.push_back(input[i]);
    }

    int max = 0;
    int it = 0;

    // UpSweep
    while (intermeadiate_results.size() > 1) {
        std::vector<int> even, odd;

        for (uint64_t i = 0; i < intermeadiate_results.size(); ++i) {
            if (i % 2 == 0)
                even.push_back(intermeadiate_results[i]);
            else
                odd.push_back(intermeadiate_results[i]);
        }

        size_t maxSize = std::max(even.size(), odd.size());
        even.resize(maxSize, 0);
        odd.resize(maxSize, 0);
        

        auto stop_cpu = std::chrono::high_resolution_clock::now();
        hostElapsedTime += (stop_cpu - start_cpu);

        if (!createDevice(params.configFile))
            return 1;

        prefixSum(even, odd, deviceoutput, maxSize);

        auto start_cpu2 = std::chrono::high_resolution_clock::now();
        it++;
        int ind = std::pow(2, it);
        intermeadiate_results = deviceoutput;
        intermeadiate_results.resize(maxSize);

        for (uint64_t i = 0; i < maxSize; ++i) {
            int index = ind * i + (ind - 1);
            input[index] = deviceoutput[i];
        }

        max = ind;

        auto stop_cpu2 = std::chrono::high_resolution_clock::now();
        hostElapsedTime += (stop_cpu2 - start_cpu2);
    }

    std::cout << "Host elapsed time before downsweep: "
              << std::fixed << std::setprecision(3)
              << hostElapsedTime.count() << " ms." << std::endl;

    auto start_cpu3 = std::chrono::high_resolution_clock::now();

    // Clear last element
    input[max - 1] = input[(max / 2) - 1];
    input[(max / 2) - 1] = 0;
    max = static_cast<int>(std::log2(max));  //eliminate the looping for first two steps
    max -= 2;
    
    
    // DownSweep
    while (max >= 0) {
        int ind2 = std::pow(2, max);
        int val = 0;
        std::vector<int> even2, odd2, result;

        for (uint64_t i = ind2 - 1; i < input.size(); i += ind2) {
            if (val % 2 == 0)
                even2.push_back(input[i]);
            else
                odd2.push_back(input[i]);
            val++;
        }
    auto stop_cpu3 = std::chrono::high_resolution_clock::now();
    hostElapsedTime += (stop_cpu3 - start_cpu3);
        //PIM kernel
        downsweep(odd2, even2, even2.size());

    auto start_cpu4 = std::chrono::high_resolution_clock::now();
        for (uint64_t i = 0; i < even2.size(); i++) {
            result.push_back(odd2[i]);
            result.push_back(even2[i]);
        }

        for (uint64_t i = 0; i < result.size(); i++) {
            int index2 = ind2 * i + (ind2 - 1);
            input[index2] = result[i];
        }
        max--;
    }

    for (uint64_t i = 0; i < host_device_merged.size(); i++) {  // Merge results 
        host_device_merged[i] += input[i];
    }
    
//Verification of Results hostresults vs deviceresults
if (params.shouldVerify)
{
    // verify result
#pragma omp parallel for
    for (uint64_t i = 0; i < len; ++i)
    {
      if (hostoutput[i] != host_device_merged[i])
      {
        std::cout << "Wrong answer for Prefixsum: " << hostoutput[i] << " != " << host_device_merged[i] << std::endl;
      }
    }
}

pimShowStats();
cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << endl;

  return 0;
}
