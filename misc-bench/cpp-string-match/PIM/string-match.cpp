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
          "\n    -c    dramsim config file"
          "\n    -i    input file containing string and key (default=generates vector with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.stringLength = 2048;
  p.keyLength = 20;
  p.configFile = nullptr;
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

  PimStatus status = pimCopyDeviceToHost(pim_obj, (void *)dst_host.data(), 0, (len+3)/4);
  assert (status == PIM_OK);

  for (auto val : dst_host) {
    std::cout << unsigned(val) << " ";
  }
  std::cout << std::endl;
}

vector<uint8_t> string_match(string& needle, string& haystack) {
  unsigned bitsPerElement = sizeof(int) * 8;

  PimObjId haystack_pim = pimAlloc(PIM_ALLOC_AUTO, haystack.size(), bitsPerElement, PIM_UINT8);
  assert(haystack_pim != -1);

  PimObjId needle_pim = pimAllocAssociated(bitsPerElement, haystack_pim, PIM_UINT8);
  assert(needle_pim != -1);

  PimStatus status = pimCopyHostToDevice((void *)haystack.c_str(), haystack_pim);
  assert (status == PIM_OK);

  // status = pimXor(needle_pim, needle_pim, needle_pim);
  // assert (status == PIM_OK);

  status = pimCopyHostToDevice((void *)needle.c_str(), needle_pim, 0, (needle.size() + 3)/4);
  assert (status == PIM_OK);

  cout << "haystack: " << endl;
  print_pim(haystack_pim, haystack.size());

  cout << "needle: " << endl;
  print_pim(needle_pim, haystack.size());
  
  return {};
}

void getString(string& str, uint64_t len) {
  str.reserve(len);
#pragma omp parallel for
  for(uint64_t i=0; i<len; ++i) {
    str.push_back('a' + (rand()%26));
  }
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Running PIM string match for string size: " << params.stringLength << ", key size: " << params.keyLength << "\n";
  string haystack, needle;
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
  
  if (!createDevice(params.configFile))
  {
    return 1;
  }

  cout << "string: " << haystack << endl;
  cout << "key: " << needle << endl;

  //TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iteration.
  string_match(needle, haystack);

  if (params.shouldVerify) 
  {
    
  }

  pimShowStats();

  return 0;
}
