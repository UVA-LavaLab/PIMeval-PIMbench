// Test: Cuda version of string match
// Copyright (c) 2025 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#include <cassert>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "string-match-utils.h"
#include "PFAC.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  char *keysInputFile;
  char *textInputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./string-match.out [options]"
          "\n"
          "\n    -k    keys input file, with each key on a seperate line (required, searches in cpp-string-match/dataset directory, note that keys are expected to be sorted by length, with smaller keys first)"
          "\n    -t    text input file to search for keys from (required, searches in cpp-string-match/dataset directory)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.keysInputFile = nullptr;
  p.textInputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:k:t:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'k':
      p.keysInputFile = optarg;
      break;
    case 't':
      p.textInputFile = optarg;
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

template <typename T>
void printVec(std::vector<T>& vec) {
  for(T elem : vec) {
    std::cout << elem << ", ";
  }
  std::cout << std::endl;
}

float string_match_gpu(std::string& needle_filename, std::string& haystack, std::vector<int>& matches) {
  PFAC_handle_t pfac_handle;
  PFAC_status_t pfac_error;
  cudaError_t cuda_error;

  pfac_error = PFAC_create(&pfac_handle);
  if ( PFAC_STATUS_SUCCESS != pfac_error ){
      std::cerr << "Pfac Error: " << PFAC_getErrorString(pfac_error) << std::endl;
      exit(1);
  }

  pfac_error = PFAC_setPlatform(pfac_handle, PFAC_PLATFORM_GPU);
  if ( PFAC_STATUS_SUCCESS != pfac_error ){
      std::cerr << "Pfac Error: " << PFAC_getErrorString(pfac_error) << std::endl;
      exit(1);
  }

  pfac_error = PFAC_readPatternFromFile(pfac_handle, needle_filename.data());
  if (PFAC_STATUS_SUCCESS != pfac_error){
    std::cerr << "Cuda Error: " << PFAC_getErrorString(pfac_error) << std::endl;
    exit(1);
  }

  pfac_error = PFAC_setTextureMode(pfac_handle, PFAC_TEXTURE_ON);
  if ( PFAC_STATUS_SUCCESS != pfac_error ){
      std::cerr << "Pfac Error: " << PFAC_getErrorString(pfac_error) << std::endl;
      exit(1);
  }

  char *gpu_text;
  int *gpu_matches;

  size_t cuda_to_alloc = (haystack.size() + sizeof(int)-1)/sizeof(int);
  cuda_error = cudaMalloc((void **) &gpu_text, cuda_to_alloc*sizeof(int));
  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << std::endl;
    exit(1);
  }
  
  cuda_error = cudaMalloc((void **) &gpu_matches, haystack.size()*sizeof(int));
  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << std::endl;
    exit(1);
  }
  
  cuda_error = cudaMemcpy(gpu_text, haystack.c_str(), haystack.size(), cudaMemcpyHostToDevice);
  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << std::endl;
    exit(1);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float timeElapsed = 0;

  cudaEventRecord(start, 0);

  pfac_error = PFAC_matchFromDevice( pfac_handle, gpu_text, haystack.size(), gpu_matches);
  if ( PFAC_STATUS_SUCCESS != pfac_error ){
      std::cerr << "Pfac Error: " << PFAC_getErrorString(pfac_error) << std::endl;
      exit(1);
  }
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeElapsed, start, stop);

  cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << std::endl;
    exit(1);
  }

  pfac_error = PFAC_destroy(pfac_handle);
  if ( PFAC_STATUS_SUCCESS != pfac_error ){
      std::cerr << "Pfac Error: " << PFAC_getErrorString(pfac_error) << std::endl;
      exit(1);
  }

  cuda_error = cudaMemcpy(matches.data(), gpu_matches, haystack.size() * sizeof(int), cudaMemcpyDeviceToHost);
  if(cuda_error != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cuda_error) << std::endl;
    exit(1);
  }

  return timeElapsed;
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  
  if(params.keysInputFile == nullptr) {
    std::cout << "Please provide a keys input file" << std::endl;
    return 1;
  }
  if(params.textInputFile == nullptr) {
    std::cout << "Please provide a text input file" << std::endl;
    return 1;
  }
  
  std::cout << "Running GPU string match for \"" << params.keysInputFile << "\" as the keys file, and \"" << params.textInputFile << "\" as the text input file\n";
  
  std::string haystack;
  std::vector<std::string> needles;
  std::vector<int> matches;

  const std::string DATASET_FOLDER_PREFIX = "./../../dataset/";

  haystack = get_text_from_file(DATASET_FOLDER_PREFIX, params.textInputFile);
  if(haystack.size() == 0) {
    std::cout << "There was an error opening the text file" << std::endl;
    return 1;
  }

  needles = get_needles_from_file(DATASET_FOLDER_PREFIX, params.keysInputFile);
  if(needles.size() == 0) {
    std::cout << "There was an error opening the keys file" << std::endl;
    return 1;
  }

  matches.resize(haystack.size());
  std::string keys_filename = DATASET_FOLDER_PREFIX + params.keysInputFile;
  float timeElapsed = string_match_gpu(keys_filename, haystack, matches);
  printf("Execution time of string match = %f ms\n", timeElapsed);

  // std::cout << "matches: ";
  // printVec(matches);

  if (params.shouldVerify) 
  {
    std::vector<int> matches_cpu;
    
    matches_cpu.resize(haystack.size(), 0);

    string_match_cpu(needles, haystack, matches_cpu);

    // verify result
    bool is_correct = true;
    #pragma omp parallel for
    for (unsigned i = 0; i < matches.size(); ++i)
    {
      if (matches[i] != matches_cpu[i])
      {
        std::cout << "Wrong answer: " << unsigned(matches[i]) << " (expected " << unsigned(matches_cpu[i]) << "), for position " << i << std::endl;
        is_correct = false;
      }
    }
    if(is_correct) {
      std::cout << "Correct for string match!" << std::endl;
    }
  }

  return 0;
}
