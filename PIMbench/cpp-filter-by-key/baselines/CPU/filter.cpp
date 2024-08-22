/**
 * @file filter.cpp
 * @brief Template for a Host Application Source File.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <stdint.h>
#include <iomanip>
#include <chrono>

#include <omp.h>
#include "../../../utilBaselines.h"

#define MY_RANGE 100

using namespace std;

/**
 * @brief cpu database filtering kernel
 */

void filterByKey(std::vector<int> &Vector, uint64_t vector_size, int key, std::vector<bool> & bitMap)
{
#pragma omp parallel for
  for (uint64_t i = 0; i < vector_size; ++i)
  {
    if (key > Vector[i])
      bitMap[i] = true;
  }
}

typedef struct Params
{
    uint64_t inVectorSize;
    int key;
    bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./filter.out [options]"
          "\n"
          "\n    -n    database size (default=65536 elements)"
          "\n    -k    value of key (default = 1)"
          "\n    -v    t = print output vector. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.inVectorSize = 65536;
  p.key = 1;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:n:k:v:")) >= 0)
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

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char **argv){

    struct Params p = getInputParams(argc, argv);

    uint64_t inVectorSize = p.inVectorSize;

    vector<int32_t> inVector(inVectorSize);
    vector<int32_t> outVector;

    std::cout << "DB element size: " << inVectorSize << std::endl;

    srand(8746219);
#pragma omp parallel for
    for (uint64_t i = 0; i < inVectorSize; i++){
        inVector[i] = rand() % MY_RANGE;
    }
    std::vector<bool> bitMap(inVectorSize, false);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // run scan
    filterByKey(inVector, inVectorSize, p.key, bitMap);
    
    // select data whose bitmap is '1'
    // Do parallel reduction with a degree of (32) by creating 32 subarrays and then combining the results in these 32 array serially.
    uint64_t nThreads = 16;
    cout << "nThreads = " << nThreads << endl;
    uint64_t outSubSize = inVectorSize/nThreads;
    vector<vector<int32_t>> outSubVector(nThreads);

#pragma omp parallel num_threads(nThreads)
    for (uint64_t i = 0; i < outSubSize; i++){
        int tid = omp_get_thread_num();
        uint64_t index = tid * outSubSize + i;
        if(index < inVectorSize){
          if(bitMap[index] == true){
            outSubVector[tid].push_back(inVector[index]);
          }
        }
    }
    int outSize = 0;

#pragma omp critical
    for (uint64_t i = 0; i < nThreads; i++){
      outSize += outSubVector[i].size();
    }
#pragma omp critical
    for (uint64_t i = 0; i < nThreads; i++){
      outVector.insert(outVector.end(), outSubVector[i].begin(), outSubVector[i].end());
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedTime = (end - start);

    if(p.shouldVerify == true){
      // for (uint64_t i = 0; i < inVectorSize; i++){
      //   cout << inVector[i] << " ";
      // }
      // cout <<endl;
      // cout << "------------------------------------" <<endl;
      // for (uint64_t i = 0; i < inVectorSize; i++){
      //   cout << bitMap[i] << " ";
      // }
      // cout <<endl;
      // cout << "------------------------------------" <<endl;
      // for (uint64_t i = 0; i < outVector.size(); i++){
      //   cout << outVector[i] << " ";
      // }
      // cout << endl;
      cout << outVector.size() <<" out of " << inVectorSize << " selected" << endl;
    }
    cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;

    return 0;
}
