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

// Params ---------------------------------------------------------------------
typedef struct Params
{
  char *keysInputFile;
  char *textInputFile;
  char *configFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./string-match.out [options]"
          "\n"
          "\n    -k    keys input file, with each key on a seperate line (required, searches in cpp-string-match/dataset directory, note that keys are expected to be sorted by length, with smaller keys first)"
          "\n    -t    text input file to search for keys from (required, searches in cpp-string-match/dataset directory)"
          "\n    -c    dramsim config file"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.keysInputFile = nullptr;
  p.textInputFile = nullptr;
  p.configFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:k:t:c:v:")) >= 0)
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
    case 'c':
      p.configFile = optarg;
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
  vector<uint32_t> dst_host;
  dst_host.resize(len, 1);

  PimStatus status = pimCopyDeviceToHost(pim_obj, (void *)dst_host.data());
  assert (status == PIM_OK);

  for (auto val : dst_host) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
}

void string_match(std::vector<std::string>& needles, std::string& haystack, std::vector<int>& matches) {

  // TODO update types when pim type conversion operation is available, currently everything uses PIM_UINT32, however this is unecessary

  uint32_t num_needles = needles.size();

  PimObjId haystack_pim = pimAlloc(PIM_ALLOC_AUTO, haystack.size(), PIM_UINT32);
  assert(haystack_pim != -1);
  
  PimObjId intermediate_pim = pimAllocAssociated(haystack_pim, PIM_UINT32);
  assert(intermediate_pim != -1);

  PimObjId final_result_pim = pimAllocAssociated(haystack_pim, PIM_UINT32);
  assert(final_result_pim != -1);

  std::vector<PimObjId> pim_individual_needle_matches;
  for(uint32_t i=0; i<num_needles; ++i) {
    pim_individual_needle_matches.push_back(pimAllocAssociated(haystack_pim, PIM_UINT32));
    assert(pim_individual_needle_matches.back() != -1);
  }

  // TODO remove when pim conversion becomes possible
  uint32_t *haystack_32bit = (uint32_t*) malloc(sizeof(uint32_t) * haystack.size());
  for(uint32_t i=0; i<haystack.size(); ++i) {
    haystack_32bit[i] = (uint32_t) haystack[i];
  }

  PimStatus status = pimCopyHostToDevice((void *)haystack_32bit, haystack_pim);
  assert (status == PIM_OK);

  // Algorithm Start
  uint64_t needles_finished = 0;

  for(uint64_t char_idx=0; needles_finished < num_needles; ++char_idx) {
    
    for(uint64_t needle_idx=0; needle_idx < num_needles; ++needle_idx) {

      if(char_idx >= needles[needle_idx].size()) {
        continue;
      }

      if(char_idx == 0) {
        status = pimEQScalar(haystack_pim, pim_individual_needle_matches[needle_idx], (uint64_t) needles[needle_idx][char_idx]);
        assert (status == PIM_OK);
      } else {
        status = pimEQScalar(haystack_pim, intermediate_pim, (uint64_t) needles[needle_idx][char_idx]);
        assert (status == PIM_OK);

        status = pimAnd(pim_individual_needle_matches[needle_idx], intermediate_pim, pim_individual_needle_matches[needle_idx]);
        assert (status == PIM_OK);
      }

      if(char_idx + 1 == needles[needle_idx].size()) {
        ++needles_finished;
      }
    }

    if(needles_finished < num_needles) {
      status = pimShiftElementsLeft(haystack_pim);
      assert (status == PIM_OK);
    }
  }

  // for(uint64_t needle_idx = 0; needle_idx < num_needles; ++needle_idx) {
  //   pimMulScalar(pim_individual_needle_matches[needle_idx], pim_individual_needle_matches[needle_idx], 1 + needle_idx);
  // }

  for(uint64_t needle_idx = 0; needle_idx < num_needles; ++needle_idx) {
    status = pimXorScalar(pim_individual_needle_matches[needle_idx], pim_individual_needle_matches[needle_idx], 1);
    assert (status == PIM_OK);

    status = pimSubScalar(pim_individual_needle_matches[needle_idx], pim_individual_needle_matches[needle_idx], 1);
    assert (status == PIM_OK);

    status = pimAndScalar(pim_individual_needle_matches[needle_idx], pim_individual_needle_matches[needle_idx], 1 + needle_idx);
    assert (status == PIM_OK);
  }

  for(uint64_t needle_idx = 1; needle_idx < num_needles; ++needle_idx) {
    status = pimMax(pim_individual_needle_matches[0], pim_individual_needle_matches[needle_idx], pim_individual_needle_matches[0]);
    assert (status == PIM_OK);
  }

  status = pimCopyDeviceToHost(pim_individual_needle_matches[0], (void *)matches.data());
  assert (status == PIM_OK);
}

void string_match_cpu(std::vector<std::string>& needles, std::string& haystack, std::vector<int>& matches) {
  for(uint64_t needle_idx = 0; needle_idx < needles.size(); ++needle_idx) {
    size_t pos = haystack.find(needles[needle_idx], 0);

    while (pos != string::npos) {
        matches[pos] = max((unsigned long) matches[pos], needle_idx + 1);
        pos = haystack.find(needles[needle_idx], pos + 1);
    }
  }
}

template <typename T>
void printVec(std::vector<T>& vec) {
  for(T elem : vec) {
    cout << elem << ", ";
  }
  std::cout << std::endl;
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
  
  std::cout << "Running PIM string match for \"" << params.keysInputFile << "\" as the keys file, and \"" << params.textInputFile << "\" as the text input file\n";
  
  std::string haystack;
  std::vector<std::string> needles;
  std::vector<int> matches;

  const std::string DATASET_FOLDER_PREFIX = "./../dataset/";

  std::string text_input_file = params.textInputFile;
  text_input_file = DATASET_FOLDER_PREFIX + text_input_file;
  
  std::ifstream text_file(text_input_file);
  if (!text_file) {
    std::cout << "Couldn't open text input file" << std::endl;
    return 1;
  }

  std::ostringstream text_file_oss;
  text_file_oss << text_file.rdbuf();
  haystack = text_file_oss.str();

  std::string keys_input_file = params.keysInputFile;
  keys_input_file = DATASET_FOLDER_PREFIX + keys_input_file;
  std::ifstream keys_file(keys_input_file);
  if (!keys_file) {
    std::cout << "Couldn't open keys input file" << std::endl;
    return 1;
  }
  std::string line;

  while (std::getline(keys_file, line)) {
      needles.push_back(line);
  }
  
  if (!createDevice(params.configFile))
  {
    return 1;
  }

  matches.resize(haystack.size(), 0);

  // //TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iteration.
  string_match(needles, haystack, matches);

  std::cout << "matches: ";
  printVec(matches);

  if (params.shouldVerify) 
  {
    std::vector<int> matches_cpu;
    
    matches_cpu.resize(haystack.size());

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

  pimShowStats();

  return 0;
}
