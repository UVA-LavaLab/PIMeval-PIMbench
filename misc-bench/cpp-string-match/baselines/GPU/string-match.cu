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

float stringMatchGpu(std::string& needleFilename, std::string& haystack, std::vector<int>& matches) {
  PFAC_handle_t pfacHandle;
  PFAC_status_t pfacError;
  cudaError_t cudaError;

  pfacError = PFAC_create(&pfacHandle);
  if ( PFAC_STATUS_SUCCESS != pfacError ){
      std::cerr << "Pfac Error: " << PFAC_getErrorString(pfacError) << std::endl;
      exit(1);
  }

  pfacError = PFAC_setPlatform(pfacHandle, PFAC_PLATFORM_GPU);
  if ( PFAC_STATUS_SUCCESS != pfacError ){
      std::cerr << "Pfac Error: " << PFAC_getErrorString(pfacError) << std::endl;
      exit(1);
  }

  pfacError = PFAC_readPatternFromFile(pfacHandle, needleFilename.data());
  if (PFAC_STATUS_SUCCESS != pfacError){
    std::cerr << "Cuda Error: " << PFAC_getErrorString(pfacError) << std::endl;
    exit(1);
  }

  pfacError = PFAC_setTextureMode(pfacHandle, PFAC_TEXTURE_ON);
  if ( PFAC_STATUS_SUCCESS != pfacError ){
      std::cerr << "Pfac Error: " << PFAC_getErrorString(pfacError) << std::endl;
      exit(1);
  }

  char *gpuText;
  int *gpuMatches;

  size_t cudaToAlloc = (haystack.size() + sizeof(int)-1)/sizeof(int);
  cudaError = cudaMalloc((void **) &gpuText, cudaToAlloc*sizeof(int));
  if(cudaError != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cudaError) << std::endl;
    exit(1);
  }
  
  cudaError = cudaMalloc((void **) &gpuMatches, haystack.size()*sizeof(int));
  if(cudaError != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cudaError) << std::endl;
    exit(1);
  }
  
  cudaError = cudaMemcpy(gpuText, haystack.c_str(), haystack.size(), cudaMemcpyHostToDevice);
  if(cudaError != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cudaError) << std::endl;
    exit(1);
  }

  float timeElapsed = 0;

  pfacError = PFAC_matchFromDevice(pfacHandle, gpuText, haystack.size(), gpuMatches, &timeElapsed);
  if (PFAC_STATUS_SUCCESS != pfacError){
      std::cerr << "Pfac Error: " << PFAC_getErrorString(pfacError) << std::endl;
      exit(1);
  }

  cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cudaError) << std::endl;
    exit(1);
  }

  pfacError = PFAC_destroy(pfacHandle);
  if ( PFAC_STATUS_SUCCESS != pfacError ){
      std::cerr << "Pfac Error: " << PFAC_getErrorString(pfacError) << std::endl;
      exit(1);
  }

  cudaError = cudaMemcpy(matches.data(), gpuMatches, haystack.size() * sizeof(int), cudaMemcpyDeviceToHost);
  if(cudaError != cudaSuccess) {
    std::cerr << "Cuda Error: " << cudaGetErrorString(cudaError) << std::endl;
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

  haystack = getTextFromFile(DATASET_FOLDER_PREFIX, params.textInputFile);
  if(haystack.size() == 0) {
    std::cout << "There was an error opening the text file" << std::endl;
    return 1;
  }

  needles = getNeedlesFromFile(DATASET_FOLDER_PREFIX, params.keysInputFile);
  if(needles.size() == 0) {
    std::cout << "There was an error opening the keys file" << std::endl;
    return 1;
  }

  matches.resize(haystack.size());
  std::string keysFilename = DATASET_FOLDER_PREFIX + params.keysInputFile;
  float timeElapsed = stringMatchGpu(keysFilename, haystack, matches);
  printf("Execution time of string match = %f ms\n", timeElapsed);

  if (params.shouldVerify) 
  {
    std::vector<int> matchesCpu;
    
    matchesCpu.resize(haystack.size(), 0);

    stringMatchCpu(needles, haystack, matchesCpu);

    // verify result
    bool ok = true;
    #pragma omp parallel for
    for (unsigned i = 0; i < matches.size(); ++i)
    {
      if (matches[i] != matchesCpu[i])
      {
        std::cout << "Wrong answer: " << unsigned(matches[i]) << " (expected " << unsigned(matchesCpu[i]) << "), for position " << i << std::endl;
        ok = false;
      }
    }
    if(ok) {
      std::cout << "Correct for string match!" << std::endl;
    }
  }

  return 0;
}
