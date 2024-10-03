/**
 * @file string-match.cpp
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
#include <iostream>
#include <vector>

#include "utilBaselines.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t stringLength;
  uint64_t keyLength;
  char *inputFile;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./string-match.out [options]"
          "\n"
          "\n    -s    string size (default=2048 elements)"
          "\n    -k    key size (default = 20 elements)"
          "\n    -i    input file containing string and key (default=generates strings with random characters)"
          "\n");
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.stringLength = 2048;
  p.keyLength = 20;
  p.inputFile = nullptr;

  int opt;
  while ((opt = getopt(argc, argv, "h:s:k:i:")) >= 0)
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
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

/**
 * @brief cpu string match kernel
 */
void string_match(string& needle, string& haystack, vector<uint8_t>& matches) {
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

  std::cout << "Running CPU string match for string size: " << params.stringLength << ", key size: " << params.keyLength << "\n";
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

  matches.resize(haystack.size());

  auto start = std::chrono::high_resolution_clock::now();
  
  string_match(needle, haystack, matches);

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsedTime = (end - start)/WARMUP;
  std::cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << std::endl;

  return 0;
}
