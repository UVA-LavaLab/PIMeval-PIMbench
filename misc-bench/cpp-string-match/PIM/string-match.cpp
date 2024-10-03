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
          "\n    -i    input file containing string and key (default=generates strings with random characters)"
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

vector<uint8_t> string_match(string& needle, string& haystack) {

  PimObjId haystack_pim = pimAlloc(PIM_ALLOC_AUTO, haystack.size(), PIM_UINT8);
  assert(haystack_pim != -1);

  PimObjId needle_pim = pimAllocAssociated(haystack_pim, PIM_UINT8);
  assert(needle_pim != -1);

  // Both can be replaced with booleans/1 bit ints
  PimObjId intermediate_pim = pimAllocAssociated(haystack_pim, PIM_UINT8);
  assert(intermediate_pim != -1);

  PimObjId result_pim = pimAllocAssociated(haystack_pim, PIM_UINT8);
  assert(result_pim != -1);

  // Algorithm Start  

  PimStatus status = pimCopyHostToDevice((void *)haystack.c_str(), haystack_pim);
  assert (status == PIM_OK);

  for(uint64_t i=0; i < needle.size(); ++i) {
    status = pimBroadcastUInt(haystack_pim, (uint64_t) needle[i]);
    assert (status == PIM_OK);

    if(i == 0) {
      status = pimEQ(haystack_pim, needle_pim, result_pim);
      assert (status == PIM_OK);
    } else {
      status = pimEQ(haystack_pim, needle_pim, intermediate_pim);
      assert (status == PIM_OK);

      status = pimAnd(result_pim, intermediate_pim, result_pim);
      assert (status == PIM_OK);
    }

    if(i+1 != needle.size()) {
      pimShiftElementsLeft(haystack_pim);
    }
  }

  vector<uint8_t> dst_host;
  dst_host.resize(haystack.size());

  status = pimCopyDeviceToHost(result_pim, (void *)dst_host.data());
  assert (status == PIM_OK);

  return dst_host;
}

// a,b,c,0,0,0,0,0,0,0,0,0
// 0,0,0,a,b,c,0,0,0,0,0,0
// Linear: s-k shifts
// 12-3=9

// a,b,c,0,0,0,0,0,0,0,0,0
// 0,0,0,a,b,c,0,0,0,0,0,0
// k + 2k
// 3 + 6 = 9

// The same


// 5-2

// abcde
// ab



// Idea 2: Hash
// a,b,c,d,e,a,b,c
// a,b,c


// a,b,c,d,e,a,b,c
// b,c,d,e,a,b,c,0
// Shift left needle.size()-1 times
// "Hash" characters
// "Hash" key (on host?)


// Gives possible match array
//     -> must get actual matches
// 1,0,0,1,0,1,0,0
//     -> getting actual matches from potential matches seems difficult, will probably need host code?
//     -> Idea 1: pull potential matches back to host, then check on host
//     -> Idea 2: take matches back to pim, put key at every match index in new pim obj, compare using strategy


// Worth looking into hash based strategies, because:
//     -> Current implementation has O(needle.size()^2) element shifts, which are slow already

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
  if (params.inputFile == nullptr)
  {
    // getString(haystack, params.stringLength);
    // getString(needle, params.keyLength);
    haystack = "abcdeabc";
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
