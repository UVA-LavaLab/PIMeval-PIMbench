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
  unsigned bitsPerElement = 8;
  unsigned bitsPerElementInt = sizeof(int) * 8;

  PimObjId haystack_pim = pimAlloc(PIM_ALLOC_AUTO, haystack.size(), bitsPerElement, PIM_UINT8);
  assert(haystack_pim != -1);

  PimObjId needle_pim = pimAllocAssociated(bitsPerElement, haystack_pim, PIM_UINT8);
  assert(needle_pim != -1);

  PimObjId needle_length_pim = pimAllocAssociated(bitsPerElement, haystack_pim, PIM_UINT8);
  assert(needle_length_pim != -1);

  cout << "needle length 1:\n";
  print_pim(needle_length_pim, haystack.size());

  PimObjId result_pim = pimAllocAssociated(bitsPerElement, haystack_pim, PIM_UINT8);
  assert(result_pim != -1);

  PimObjId tmp_32 = pimAllocAssociated(bitsPerElementInt, haystack_pim, PIM_UINT32);
  assert(tmp_32 != -1);

  PimStatus status = pimCopyHostToDevice((void *)haystack.c_str(), haystack_pim);
  assert (status == PIM_OK);

  status = pimXor(needle_length_pim, needle_length_pim, needle_length_pim);
  assert (status == PIM_OK);

  cout << "needle length 2:\n";
  print_pim(needle_length_pim, haystack.size());

  int needle_copies = haystack.size() / needle.size();
  uint8_t needle_length_plus = 1 + needle.size();
  for(uint64_t i=0; i<needle_copies; ++i) {
    status = pimCopyHostToDevice((void *)needle.c_str(), needle_pim, i*needle.size(), (i+1)*needle.size());
    assert (status == PIM_OK);

    cout << "needle length:\n";
    print_pim(needle_length_pim, haystack.size());

    status = pimCopyHostToDevice((void *)(&needle_length_plus), needle_length_pim, i*needle.size(), i*needle.size()+1);
    assert (status == PIM_OK);
  }

  cout << "needle length:\n";
  print_pim(needle_length_pim, haystack.size());

  // Can be replaced with 1 bit in future
  PimObjId curr_match = pimAllocAssociated(bitsPerElement, haystack_pim, PIM_UINT8);
  assert(curr_match != -1);

  PimObjId match_accumulator = pimAllocAssociated(bitsPerElement, haystack_pim, PIM_UINT8);
  assert(match_accumulator != -1);

  for(uint64_t i=0; i<needle.size(); ++i) {
    status = pimEQ(haystack_pim, needle_pim, curr_match);
    assert (status == PIM_OK);
    // cout << "after eq";

    status = pimAddScalar(curr_match, match_accumulator, 1);
    assert (status == PIM_OK);

    for(uint64_t i=0; i<needle.size()-1; ++i) {
      pimShiftElementsLeft(curr_match);
      pimAdd(match_accumulator, curr_match, match_accumulator);
    }

    // cout << "match accumulator after loop:\n";
    // print_pim(match_accumulator, haystack.size());
    // cout << "needle length before match:\n";
    // print_pim(needle_length_pim, haystack.size());

    status = pimEQ(match_accumulator, needle_length_pim, match_accumulator);
    assert (status == PIM_OK);

    // cout << "match accumulator after eq:\n";
    // print_pim(match_accumulator, haystack.size());

    status = pimOr(match_accumulator, result_pim, result_pim);
    assert (status == PIM_OK);

    if(i+1 != needle.size()) {
      pimShiftElementsRight(needle_pim);
      pimShiftElementsRight(needle_length_pim);
    }
  }
  
  // cout << "needle:\n";
  // print_pim(needle_pim, haystack.size());

  // cout << "needle length:\n";
  // print_pim(needle_length_pim, haystack.size());

  // cout << "match accumulator:\n";
  // print_pim(match_accumulator, haystack.size());

  cout << "result pim:\n";
  print_pim(result_pim, haystack.size());
  
  return {};
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
