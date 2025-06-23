// Radix Sort implementation on bitSIMD
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include "util.h"
#include <iostream>
#include <vector>
#include <getopt.h>
#include <cstdlib>
#include <time.h> 
#include <algorithm>
#include <chrono>
using namespace std;

std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();
std::chrono::duration<double, std::milli> hostElapsedTimePref = std::chrono::duration<double, std::milli>::zero();

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t arraySize;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;


void usage()
{
  fprintf(stderr,
          "\n    Usage:  ./radix-sort.out [options]"
          "\n"
          "\n    -n    array size (default=2048 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing the array of value to be sort (default=generates datapoints with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=true)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.arraySize = 2048;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = true;

  int opt;
  while ((opt = getopt(argc, argv, "h:n:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'n':
      p.arraySize = strtoull(optarg, NULL, 0);
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

void runPrefixSum(uint64_t vectorLength, std::vector<int64_t> &src, std::vector<int64_t> &dst) {
  PimObjId srcObj = pimAlloc(PIM_ALLOC_AUTO, vectorLength, PIM_INT64);
  if (srcObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice((void *)src.data(), srcObj);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimDeviceProperties deviceProp;
  status = pimGetDeviceProperties(&deviceProp);
  if (deviceProp.isHLayoutDevice) {
    PimObjId dstObj = pimAllocAssociated(srcObj, PIM_INT64);
    if (dstObj == -1)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    status = pimPrefixSum(srcObj, dstObj);
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
    pimFree(dstObj);
  } else {
    std::vector <int64_t> tempVec = src;
    PimObjId maskObj = pimAllocAssociated(srcObj, PIM_INT64);
    if (maskObj == -1)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    std::vector<int64_t> maskVec (vectorLength, 0);
    for (uint64_t i = 0; (uint64_t)(1 << i) < vectorLength; ++i) {
      auto start_cpu = std::chrono::high_resolution_clock::now();
      #pragma omp parallel for
      for (uint64_t j = 0; j < vectorLength; ++j) {
        if (j < (uint64_t)(1 << i)) maskVec[j] = 0;
        else maskVec[j] = tempVec[j - (1 << i)];
      }
      auto stop_cpu = std::chrono::high_resolution_clock::now();
      hostElapsedTimePref += (stop_cpu - start_cpu);
      status = pimCopyHostToDevice((void *)maskVec.data(), maskObj);
      if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        return;
      }
      status = pimAdd(srcObj, maskObj, srcObj);
      status = pimCopyDeviceToHost(srcObj, tempVec.data());
      if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        return;
      }
    }
    dst = tempVec;
    pimFree(maskObj);
  }
  pimFree(srcObj);
}

int main(int argc, char *argv[])
{
    struct Params params = getInputParams(argc, argv);
    std::cout << "PIM test: Radix Sort" << std::endl;

    if (!createDevice(params.configFile)){
        return 1;
    }

    uint64_t numElements = params.arraySize;
    //parameters that can be changed to explore design space
    unsigned bitsPerElement = 32;
    unsigned radix_bits = 8;
    unsigned num_passes = bitsPerElement / radix_bits;
    unsigned radix = 1 << radix_bits;

    //Allocating Pimobj for all the iterations
    std::vector<PimObjId> src_obj(num_passes);
    std::vector<PimObjId> compare_obj(num_passes);
    std::vector<PimObjId> compare_results_obj(num_passes);

    for(unsigned i = 0; i < num_passes; i++){
        src_obj[i] = pimAlloc(PIM_ALLOC_AUTO, numElements, PIM_INT32);
        if (src_obj[i] == -1) {
            std::cout << "Abort" << std::endl;
            return 1;
        }
    }
    for(unsigned i = 0; i < num_passes; i++){
        compare_obj[i] = pimAllocAssociated(src_obj[i], PIM_INT32);
        if (compare_obj[i] == -1) {
            std::cout << "Abort" << std::endl;
            return 1;
        }
    }
    for(unsigned i = 0; i < num_passes; i++){
        compare_results_obj[i] = pimAllocAssociated(src_obj[i], PIM_BOOL);
        if (compare_results_obj[i] == -1) {
            std::cout << "Abort" << std::endl;
            return 1;
        }
    }

    //vectore for host use
    std::vector<int> src1(numElements);
    std::vector<int> dest(numElements);
    //array used to check result
    std::vector<int> sorted_array(numElements);
    //counting table in host
    std::vector<int64_t> count_table(radix);
    
    //Assign random initial values to the input array
    getVector(numElements, src1);

    sorted_array = src1;

    unsigned mask = 0x000000FF;

    //Outer iteration of radix sort, each iteration perform a counting sort
    for (unsigned i = 0; i < num_passes; i++){
        std::fill(count_table.begin(), count_table.end(), 0);

        //Create a slice of 'radix_bits' of the input array and only copy that array to bitSIMD
        std::vector<unsigned> src1_slice(numElements);  //shoud be an array of 8-bit elements if radix_bits=8
        for (unsigned j = 0; j < numElements; j++){
            src1_slice[j] = src1[j] & mask; //get the slices of all elements in the array
        }

        PimStatus status = pimCopyHostToDevice((void*)src1_slice.data(), src_obj[i]);
        if (status != PIM_OK) {
            std::cout << "Abort" << std::endl;
            return 1;
        }

        //loop to count the occurance of all the possible number in sliced bit
        for (unsigned j = 0; j < radix; j++){
            unsigned brdcast_value = (j << (i * radix_bits)) & mask;

            status = pimEQScalar(src_obj[i], compare_results_obj[i], brdcast_value);
            if (status != PIM_OK) {
                std::cout << "Abort" << std::endl;
                return 1;
            }

            status = pimRedSum(compare_results_obj[i], static_cast<void*>(&count_table[j]),  0, numElements);
            if (status != PIM_OK) {
                std::cout << "Abort" << std::endl;
                return 1;
            }
        }

        runPrefixSum(radix, count_table, count_table);
        //Assuming the BitSIMD support 8 bits EQ, so CPU doesn't need to creat slice
        //host do prefix scan on the counting table
        // for (unsigned j = 1; j < radix; j++){
        //     count_table[j] = count_table[j] + count_table[j-1];
        // }

        //host perform reording on temp_array and copy it to src1
        std::vector<int> temp_array(numElements);


        auto start_cpu = std::chrono::high_resolution_clock::now();
        unsigned shiftMask = i * radix_bits;
        for(int j = (int)(numElements - 1); j >= 0; j--){
            unsigned element_num = (src1[j] & mask) >> shiftMask; //get the element number in the counting table
            temp_array[count_table[element_num]-1] = src1[j];
            count_table[element_num]--;
        }
        auto stop_cpu = std::chrono::high_resolution_clock::now();
        hostElapsedTime += (stop_cpu - start_cpu);
        
        src1 = temp_array;
        //shift mask bit for next iteration
        mask = mask << radix_bits;
    }

    // !check results and print it like km
    pimShowStats();

    if (params.shouldVerify){
        bool ok = true;
        std::sort(sorted_array.begin(), sorted_array.end());
        if(sorted_array != src1){
            std::cout << "Wrong answer!" << std::endl;
            ok = false;
        }
        if (ok) {
            std::cout << "All correct!" << std::endl;
        }
    }

    cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << endl;
    cout << "Host elapsed time for Prefix Sum: " << std::fixed << std::setprecision(3) << hostElapsedTimePref.count() << " ms." << endl;
    return 0;
}
