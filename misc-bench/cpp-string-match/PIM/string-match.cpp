// Test: C++ version of string match
// Copyright (c) 2024 University of Virginia
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

#include "../../util.h"
#include "libpimeval.h"

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t stringLength;
  uint64_t keyLength;
  uint64_t numKeys;
  char *configFile;
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
          "\n    -n    number of keys (default = 4 keys)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing string and keys (default=generates strings with random characters)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.stringLength = 2048;
  p.keyLength = 20;
  p.numKeys = 4;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:s:k:n:c:i:v:")) >= 0)
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
    case 'n':
      p.numKeys = strtoull(optarg, NULL, 0);
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

void print_pim(PimObjId pim_obj, uint64_t len) {
  vector<uint8_t> dst_host;
  dst_host.resize(len, 1);

  PimStatus status = pimCopyDeviceToHost(pim_obj, (void *)dst_host.data());
  assert (status == PIM_OK);

  for (auto val : dst_host) {
    std::cout << unsigned(val) << " ";
  }
  std::cout << std::endl;
}

void print_pim_int(PimObjId pim_obj, uint64_t len) {
  vector<int> dst_host;
  dst_host.resize(len, 1);

  PimStatus status = pimCopyDeviceToHost(pim_obj, (void *)dst_host.data());
  assert (status == PIM_OK);

  for (auto val : dst_host) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
}

void string_match(string& needle, string& haystack, vector<uint8_t>& matches) {

  PimObjId haystack_pim = pimAlloc(PIM_ALLOC_AUTO, haystack.size(), PIM_UINT8);
  assert(haystack_pim != -1);

  // Both can be replaced with booleans/1 bit ints
  PimObjId intermediate_pim = pimAllocAssociated(haystack_pim, PIM_UINT8);
  assert(intermediate_pim != -1);

  PimObjId result_pim = pimAllocAssociated(haystack_pim, PIM_UINT8);
  assert(result_pim != -1);

  // Algorithm Start  

  PimStatus status = pimCopyHostToDevice((void *)haystack.c_str(), haystack_pim);
  assert (status == PIM_OK);

  for(uint64_t i=0; i < needle.size(); ++i) {
    if(i == 0) {
      status = pimEQScalar(haystack_pim, result_pim, (uint64_t) needle[i]);
      assert (status == PIM_OK);
    } else {
      status = pimEQScalar(haystack_pim, intermediate_pim, (uint64_t) needle[i]);
      assert (status == PIM_OK);

      status = pimAnd(result_pim, intermediate_pim, result_pim);
      assert (status == PIM_OK);
    }

    if(i+1 != needle.size()) {
      pimShiftElementsLeft(haystack_pim);
    }
  }

  status = pimCopyDeviceToHost(result_pim, (void *)matches.data());
  assert (status == PIM_OK);
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

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Running PIM string match for string size: " << params.stringLength << ", key size: " << params.keyLength << "\n";
  string haystack, needle;
  vector<uint8_t> matches;
  if (params.inputFile == nullptr)
  {
    // getString(haystack, params.stringLength);
    // getString(needle, params.keyLength);
    haystack = "dabcd";
    needle = "abc";
  } 
  else 
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 1;
  }
  
  if (!createDevice(params.configFile))
  {
    return 1;
  }

  matches.resize(haystack.size());
  //TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iteration.
  string_match(needle, haystack, matches);

  if (params.shouldVerify) 
  {
    vector<uint8_t> matches_cpu;
    matches_cpu.resize(haystack.size());
    string_match_cpu(needle, haystack, matches_cpu);

    // verify result
    bool is_correct = true;
    #pragma omp parallel for
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

  pimShowStats();

  return 0;
}
