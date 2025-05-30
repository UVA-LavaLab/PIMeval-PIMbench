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

void prefixSum(vector<int> &right, vector<int> &left,uint64_t len)
{
 
  PimObjId rightObj = pimAlloc(PIM_ALLOC_AUTO, len, PIM_INT32);
  if (rightObj == -1)
  {
    std::cerr << "Abort: Failed to allocate memory on PIM." << std::endl;
    return;
  }
  PimStatus status = pimCopyHostToDevice((void *)right.data(), rightObj);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy data to PIM." << std::endl;
    return;
  }
  

  PimObjId leftObj = pimAllocAssociated(rightObj , PIM_INT32);
  if (leftObj == -1)
  {
    std::cerr << "Abort: Failed to allocate memory on PIM." << std::endl;
    return;
  }
  status = pimCopyHostToDevice((void *)left.data(), leftObj);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy data to PIM." << std::endl;
  } 


  //PIM Add
  status = pimAdd(rightObj, leftObj, rightObj);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to perform PIM addition." << std::endl;
    return;
  }

  //Copy results back to Host
  status = pimCopyDeviceToHost(rightObj, (void *)right.data());
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy prefix sum result from PIM." << std::endl;
    return;
  }

  // Clean up PIM objects
  pimFree(rightObj);
  pimFree(leftObj);
}


void downsweep(vector<int> &left, vector<int> &right, uint64_t len)
{
  PimObjId rightObj = pimAlloc(PIM_ALLOC_AUTO, len, PIM_INT32);
  if (rightObj == -1)
  {
    std::cerr << "Abort: Failed to allocate memory on PIM." << std::endl;
    return;
  }
  PimStatus status = pimCopyHostToDevice((void *)right.data(), rightObj);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy data to PIM." << std::endl;
    return;
  }

  PimObjId leftObj = pimAllocAssociated(rightObj, PIM_INT32);
  if (rightObj == -1)
  {
    std::cerr << "Abort: Failed to allocate memory on PIM." << std::endl;
    return;
  }
  status = pimCopyHostToDevice((void *)left.data(), leftObj);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy data to PIM." << std::endl;
  } 

  //PIM Add
  status = pimAdd(rightObj, leftObj, rightObj);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to perform PIM addition." << std::endl;
    return;
  }

  //Copy results back to Host
  status = pimCopyDeviceToHost(rightObj, (void *)right.data());
  if (status != PIM_OK)
  {
    std::cerr << "Abort: Failed to copy prefix sum result from PIM." << std::endl;
    return;
  } 

  //Clean up PIM objects
  pimFree(rightObj);
  pimFree(leftObj);
}

bool isPowerofTwo(int n) {
    if (n <= 0)
        return false;

    int logValue = log2(n);
    return pow(2, logValue) == n;
}

int nextPowerOfTwo(int n) {
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}

int main(int argc, char *argv[]) {
    struct Params params = getInputParams(argc, argv);
    std::vector<int> input;

    if (params.inputFile == nullptr) {
        getVector(params.vectorLength, input);
    } else {
        std::cout << "Reading from input file is not implemented yet." << std::endl;
        return 1;
    }

    int len = input.size();
    if (!isPowerofTwo(len)) {
      int result = nextPowerOfTwo(len);
      int padding = result + len;
      input.resize(padding, 0);
      len = input.size();
    } else {
        std::cout << "Input size is already a power of two. " << std::endl;
    }

    std::vector<int> host_device_merged(len);
    std::vector<int> hostoutput(len);
    
    auto start_cpu = std::chrono::high_resolution_clock::now();
    hostoutput[0] = input[0];
    for (uint64_t i = 0; i < input.size(); i++) {
      hostoutput[i] = input[i]+ hostoutput[i-1];
      host_device_merged[i]=input[i];
    }

    int max = 0;
    int it = 0;
    std::vector<int> right(len);
    std::vector<int> left(len);   
    size_t iterations = input.size();

// UpSweep
while (iterations > 1) {
    size_t num_right = (iterations + 1) / 2;
    size_t num_left = iterations / 2;
    int position = std::pow(2, it);

    right.resize(num_right);
    left.resize(num_left);

    size_t right_idx = 0, left_idx = 0;

    for (size_t i = 0; i < right.size(); i++) {
      
        size_t running_indexex_right = position * (2 * i) + (position - 1);
        size_t running_indexex_left  = position * (2 * i + 1) + (position - 1);

        right[right_idx++] = input[running_indexex_right];
        left[left_idx++]   = input[running_indexex_left];
    }

        
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    hostElapsedTime += (stop_cpu - start_cpu);

    if (!createDevice(params.configFile))
        return 1;

    prefixSum(right, left, right.size());

    auto start_cpu2 = std::chrono::high_resolution_clock::now();
    it++;
    int running_index = std::pow(2, it);
    iterations = right.size();
  
    for (uint64_t i = 0; i < right.size(); ++i) {
        int running_indexex = running_index * i + (running_index - 1);
        input[running_indexex] = right[i];
    }

    max = running_index;

    auto stop_cpu2 = std::chrono::high_resolution_clock::now();
    hostElapsedTime += (stop_cpu2 - start_cpu2);
}

auto start_cpu3 = std::chrono::high_resolution_clock::now();

// DownSweep
input[max - 1] = input[(max / 2) - 1];  // Clear last element
input[(max / 2) - 1] = 0;
max = static_cast<int>(std::log2(max));  // eliminate the looping for first two steps
max -= 2;

while (max >= 0) {
    int position = std::pow(2, max);
    int val = 0;
    size_t partitions = 0;
    
    for (uint64_t i = position - 1; i < input.size(); i += position)
    partitions++;
    size_t num_right = (partitions + 1) / 2;
    size_t num_left = partitions / 2;

    right.resize(num_right);
    left.resize(num_left);
    size_t right_idx = 0, left_idx = 0;

    for (uint64_t i = position - 1; i < input.size(); i+= position) {

        if (val % 2 == 0)
            right[right_idx++] = input[i];
        else
            left[left_idx++] = input[i];
        val++;
    }

    auto stop_cpu3 = std::chrono::high_resolution_clock::now();
    hostElapsedTime += (stop_cpu3 - start_cpu3);

    // PIM kernel
    downsweep(left, right, right.size());

    for (size_t i = 0; i < right.size(); i++) {

        size_t running_indexex_left = position * (2 * i) + (position - 1);
        size_t running_indexex_right = position * (2 * i + 1) + (position - 1);

        if (i < left.size())
            input[running_indexex_left] = left[i];
            input[running_indexex_right] = right[i];
    }

    max--;
}

for (uint64_t i = 0; i < host_device_merged.size(); i++) {            // Merge results 
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
