// A trivial database filtering implementation on bitSIMD
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include "../../util.h"
#include <iostream>
#include <vector>
#include <getopt.h>
#include <cstdlib>
#include <time.h> 
#include <algorithm>
#include <chrono>
#include <bitset>

#define MY_RANGE 1000

using namespace std;


typedef struct Params
{
    uint64_t inVectorSize;
    uint64_t key;
    char *configFile;
    char *inputFile;
    bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\n    Usage:  ./filter.out [options]"
          "\n"
          "\n    -n    database size (default=2048 elements)"
          "\n    -k    value of key (default = 1)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing the database (default=generates datapoints with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=true)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.inVectorSize = 2048;
  p.key = 10;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = true;

  int opt;
  while ((opt = getopt(argc, argv, "h:n:k:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'n':
      p.inVectorSize = strtoull(optarg, NULL, 0);
      break;
    case 'k':
      p.key = strtoull(optarg, NULL, 0);
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


int main(int argc, char **argv){

    struct Params p = getInputParams(argc, argv);
    std::cout << "PIM test: database-filtering" << std::endl;
    uint64_t inVectorSize;
    int key;
    
    if (p.inputFile == nullptr){
        inVectorSize = p.inVectorSize;
        key = p.key;
    }
    else{
        std::cout << "Reading from input file is not implemented yet." << std::endl;
        return 1;
    }
    bool* bitMapTemp = new bool[inVectorSize];
    vector<uint64_t> inVector(inVectorSize);

    std::cout << "DB element size: " << inVectorSize << std::endl;

    srand((unsigned)time(NULL));
    for (uint64_t i = 0; i < inVectorSize; i++){
        inVector[i] = rand() % MY_RANGE;
    }

    std::vector<uint64_t> bitMap(inVectorSize, 0);
    std::vector<int> bitMapHost(inVectorSize, 0);

    if (!createDevice(p.configFile)){
        return 1;
    }

    //PIM parameters

    PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, inVector.size(), PIM_UINT64);
    if (srcObj1 == -1){
        std::cout << "Abort" << std::endl;
        return 1;
    }
    PimObjId srcObj2 = pimAllocAssociated(srcObj1, PIM_UINT64);
    if (srcObj2 == -1){
        std::cout << "Abort" << std::endl;
        return 1;
    }

    PimStatus status = pimCopyHostToDevice((void *)inVector.data(), srcObj1);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        return 1;
    }

    status = pimLTScalar(srcObj1, srcObj2, key);
    if (status != PIM_OK) {
        std::cout << "Abort" << std::endl;
        return 1;
    }

    status = pimCopyDeviceToHost(srcObj2, (void *)bitMap.data());
    if (status != PIM_OK){
      std::cout << "Abort" << std::endl;
    }
    pimShowStats();

    pimFree(srcObj1);
    pimFree(srcObj2);
    
    uint64_t dummyVectorSize = 1073741824;
    vector<int> dummyVector1(dummyVectorSize, 0);

    auto start_cpu = std::chrono::high_resolution_clock::now();

    for (uint64_t j = 0; j < dummyVectorSize; j++){
      dummyVector1[j] += rand() % MY_RANGE;
    }
    cout << "Cache flushed!" << endl;

    for (uint64_t i = 0; i < inVectorSize; i++){
      if(bitMap[i] == 1){
        bitMapTemp[i] = true;
      }
    }

    uint64_t buffer_in_CPU = 0;
    uint64_t buffer_in_CPU1 = 0;
    uint64_t buffer_in_CPU2 = 0;
    uint64_t buffer_in_CPU3 = 0;
    uint64_t buffer_in_CPU4 = 0;
    uint64_t buffer_in_CPU5 = 0;
    uint64_t buffer_in_CPU6 = 0;
    uint64_t buffer_in_CPU7 = 0;
    uint64_t buffer_in_CPU8 = 0;
    uint64_t selectedNum1 = 0;
    uint64_t selectedNum2 = 0;
    uint64_t selectedNum3 = 0;
    uint64_t selectedNum4 = 0;
    uint64_t selectedNum5 = 0;
    uint64_t selectedNum6 = 0;
    uint64_t selectedNum7 = 0;
    uint64_t selectedNum8 = 0;

    start_cpu = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < inVectorSize; i+=8){
      if(bitMapTemp[i + 0] == true){
        buffer_in_CPU1 += inVector[i + 0];
        selectedNum1++;
      }
      if(bitMapTemp[i + 1] == true){
        buffer_in_CPU2 += inVector[i + 1];
        selectedNum2++;
      }
      if(bitMapTemp[i + 2] == true){
        buffer_in_CPU3 += inVector[i + 2];
        selectedNum3++;
      }
      if(bitMapTemp[i + 3] == true){
        buffer_in_CPU4 += inVector[i + 3];
        selectedNum4++;
      }
      if(bitMapTemp[i + 4] == true){
        buffer_in_CPU5 += inVector[i + 4];
        selectedNum5++;
      }
      if(bitMapTemp[i + 5] == true){
        buffer_in_CPU6 += inVector[i + 5];
        selectedNum6++;
      }
      if(bitMapTemp[i + 6] == true){
        buffer_in_CPU7 += inVector[i + 6];
        selectedNum7++;
      }
      if(bitMapTemp[i + 7] == true){
        buffer_in_CPU8 += inVector[i + 7];
        selectedNum8++;
      }
    }
    uint64_t outSize = selectedNum1 + selectedNum2 + selectedNum3 + selectedNum4 + selectedNum5 + selectedNum6 + selectedNum7 + selectedNum8;

    auto stop_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> hostElapsedTime = (stop_cpu - start_cpu);
    if (p.shouldVerify){
        cout << outSize <<" out of " << inVectorSize << " selected" << endl;
    }
    cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << endl;
    return 0;
}
