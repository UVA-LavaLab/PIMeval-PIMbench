/* File:     scale.cu
 * Purpose:  Implement vector scaling on a GPU
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
#include <iostream>
#include <vector>

#include "utilBaselines.h"

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t stringLength;
  uint64_t keyLength;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./string-match.out [options]"
          "\n"
          "\n    -s    string size (default=2048 elements)"
          "\n    -k    key size (default = 20 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing string and key (default=generates strings with random characters)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.stringLength = 2048;
  p.keyLength = 20;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:s:k:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 's':
      p.stringLength = strtoull(optarg, NULL, 0);
      break;
    case 'k':
      p.keyLength = strtoull(optarg, NULL, 0);
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

/**
 * @brief gpu string match kernel
 */
__global__ void string_match(char* haystack, size_t haystack_len, char* needle, size_t needle_len, uint8_t* matches) {
  size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < haystack_len - needle_len + 1) {
    matches[idx] = 1;
    for (int i = 0; i < needle_len; ++i) {
      if (haystack[idx + i] != needle[i]) {
          matches[idx] = 0;
      }
    }
  }
}

void string_match_cpu(string& needle, string& haystack, vector<uint8_t>& matches) {
  size_t pos = haystack.find(needle, 0);

  if (pos == string::npos) {
    return;
  }

  while (pos != string::npos) {
      matches[pos] = 1;
      pos = haystack.find(needle, pos + 1);
  }
}

void getString(string& str, uint64_t len) {
  str.resize(len);
#pragma omp parallel for
  for(uint64_t i=0; i<len; ++i) {
    str[i] = 'a' + (rand()%26);
  }
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char **argv)
{
  struct Params params = input_params(argc, argv);
  std::cout << "Running PIM string match for string size: " << params.stringLength << ", key size: " << params.keyLength << "\n";
  string haystack, needle;
  vector<uint8_t> matches;

  if (params.inputFile == nullptr)
  {
    getString(haystack, params.stringLength);
    getString(needle, params.keyLength);
  } 
  else 
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 1;
  }

  char* gpu_haystack;
  char* gpu_needle;
  uint8_t* gpu_matches;
  matches.resize(params.stringLength);

  size_t haystack_sz = sizeof(char)*params.stringLength;
  size_t needle_sz = sizeof(char)*params.keyLength;
  size_t matches_sz = sizeof(uint8_t)*params.stringLength;

  cudaError_t cuda_error;
  cuda_error = cudaMalloc((void**)&gpu_haystack, haystack_sz);

  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
    exit(1);
  }

  cuda_error = cudaMemcpy(gpu_haystack, haystack.c_str(), haystack_sz, cudaMemcpyHostToDevice);

  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
    exit(1);
  }

  cuda_error = cudaMalloc((void**)&gpu_needle, needle_sz);

  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
    exit(1);
  }

  cuda_error = cudaMemcpy(gpu_needle, needle.c_str(), needle_sz, cudaMemcpyHostToDevice);

  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
    exit(1);
  }

  cuda_error = cudaMalloc((void**)&gpu_matches, matches_sz);

  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
    exit(1);
  }

  cuda_error = cudaMemset(gpu_matches, 0, matches_sz);

  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
    exit(1);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float timeElapsed = 0;

  cudaEventRecord(start, 0);

  string_match<<<(params.stringLength + 1023) / 1024, 1024>>>(gpu_haystack, params.stringLength, gpu_needle, params.keyLength, gpu_matches);
  
  cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess)
  {
      std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
      exit(1);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeElapsed, start, stop);

  printf("Execution time of string match = %f ms\n", timeElapsed);

  cuda_error = cudaMemcpy(matches.data(), gpu_matches, matches_sz, cudaMemcpyDeviceToHost);
  if (cuda_error != cudaSuccess)
  {
      cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << "\n";
      exit(1);
  }
  cudaFree(gpu_haystack);
  cudaFree(gpu_needle);
  cudaFree(gpu_matches);

  if (params.shouldVerify) 
  {
    vector<uint8_t> matches_cpu;
    matches_cpu.resize(haystack.size());
    string_match_cpu(needle, haystack, matches_cpu);

    // verify result
    // #pragma omp parallel for
    bool is_correct = true;
    for (unsigned i = 0; i < matches.size(); ++i)
    {
      if (matches[i] != matches_cpu[i])
      {
        std::cout << "Wrong answer: " << unsigned(matches[i]) << " (expected " << unsigned(matches_cpu[i]) << "), at index: " << i << std::endl;
        is_correct = false;
      }
    }
    if(is_correct) {
      std::cout << "Correct for string match!" << std::endl;
    }
  }

  return 0;
}
