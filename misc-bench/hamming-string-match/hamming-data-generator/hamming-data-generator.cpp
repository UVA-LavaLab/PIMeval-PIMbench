// Test: Data Generator for hamming string matching
// Copyright (c) 2025 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <unistd.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <iterator>
#include <random>
#include <iomanip>
#include <ios>
#include <vector>
#include <unordered_set>
#include <string>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "utilStringMatch.h"

typedef struct Params
{
  char *outputFile;
  size_t textLen;
  size_t numKeys;
  size_t minKeyLen;
  size_t maxKeyLen;
  size_t maxHammingDistance;
  uint8_t keyFrequency;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./data-generator.out [options]"
          "\n"
          "\n    -o    output folder name, stores output in dataset/[name]/text.txt, dataset/[name]/keys.txt, and dataset/[name]/maxHammingDistance.txt (required)"
          "\n    -l    length of text to match (default=10,000)"
          "\n    -n    number of keys (default=5)"
          "\n    -m    minimum key length (default = 1)"
          "\n    -x    maximum key length (default = 10)"
          "\n    -d    maximum hamming distance (default = 3)"
          "\n    -f    approximate frequency of keys in generated text, 0=no matches, 100=all keys (default = 50)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.outputFile = nullptr;
  p.textLen = 10000;
  p.numKeys = 5;
  p.minKeyLen = 1;
  p.maxKeyLen = 10;
  p.maxHammingDistance = 3;
  p.keyFrequency = 50;

  int opt;
  while ((opt = getopt(argc, argv, "h:o:l:n:m:x:d:f:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'o':
      p.outputFile = optarg;
      break;
    case 'l':
      p.textLen = strtoull(optarg, NULL, 0);
      break;
    case 'n':
      p.numKeys = strtoull(optarg, NULL, 0);
      break;
    case 'm':
      p.minKeyLen = strtoull(optarg, NULL, 0);
      break;
    case 'x':
      p.maxKeyLen = strtoull(optarg, NULL, 0);
      break;
    case 'd':
      p.maxHammingDistance = strtoull(optarg, NULL, 0);
      break;
    case 'f':
      p.keyFrequency = static_cast<uint8_t>(strtoull(optarg, NULL, 0));
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

int main(int argc, char* argv[])
{

  struct Params params = getInputParams(argc, argv);
  std::cout << "Generating Data..." << std::endl;

  generateStringMatchData(params.outputFile, params.textLen,
    params.numKeys, params.minKeyLen,
    params.maxKeyLen, params.keyFrequency,
    true, params.maxHammingDistance);
  
  return 0;
}
