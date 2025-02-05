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
#include <algorithm>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../../util.h"
#include "libpimeval.h"
#include "string-match-utils.h"

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

std::vector<std::vector<std::vector<size_t>>> string_match_precompute_table(std::vector<std::string>& needles, uint64_t num_rows, bool is_horizontal) {
  std::vector<std::vector<std::vector<size_t>>> result_table;
  
  uint64_t max_needles_per_iteration = is_horizontal ? num_rows - 2 : (num_rows>>5) - 2;
  uint64_t num_iterations;
  if(needles.size() <= max_needles_per_iteration) {
    num_iterations = 1;
  } else {
    uint64_t needles_after_first_iteration = needles.size() - max_needles_per_iteration;
    // Can do 1 more needle in first iteration than in later iterations
    // During the first iteration we can use the final result array as an individual result array in the first iteration
    num_iterations = 1 + ((needles_after_first_iteration + max_needles_per_iteration - 2) / (max_needles_per_iteration - 1));
  }

  result_table.resize(num_iterations);

  uint64_t needles_done = 0;

  for(uint64_t iter=0; iter<num_iterations; ++iter) {
    uint64_t needles_this_iteration;
    if(iter == 0) {
      needles_this_iteration = min(max_needles_per_iteration, needles.size());
    } else if(iter+1 == num_iterations) {
      needles_this_iteration = needles.size() - needles_done;
    } else {
      needles_this_iteration = max_needles_per_iteration - 1;
    }

    // std::cout << "Iteration number: " << iter << ", will do " << needles_this_iteration << " needles this iteration." << std::endl;
    // Range: [needles_done, needles_done + needles_this_iteration - 1]
    uint64_t first_needle_this_iteration = needles_done;
    uint64_t last_needle_this_iteration = first_needle_this_iteration + needles_this_iteration - 1;
    uint64_t longest_needle_this_iteration = needles[last_needle_this_iteration].size();
    // As we iterate through character indices for the needles in this iteration, there may be some needles that are shorter than the current character
    // Skip checking them by keeping track of the shortest needle that is long enough to have the current character
    uint64_t current_start_needle = first_needle_this_iteration;
    result_table[iter].resize(longest_needle_this_iteration);
    for(uint64_t char_ind = 0; char_ind < longest_needle_this_iteration; ++char_ind) {
      while(needles[current_start_needle].size() <= char_ind) {
        ++current_start_needle;
      }
      std::vector<size_t>& current_table_row = result_table[iter][char_ind];
      current_table_row.resize(1 + last_needle_this_iteration - current_start_needle);
      // Sort needles [current_start_needle, last_needle_this_iteration] on char_ind
      
      std::iota(current_table_row.begin(), current_table_row.end(), current_start_needle);
      std::sort(current_table_row.begin(), current_table_row.end(), [&needles, &char_ind](auto& l, auto& r) {
        return needles[l][char_ind] < needles[r][char_ind];
      });
    }

    needles_done += needles_this_iteration;
  }

  return result_table;
}

void print_table(std::vector<std::string>& needles, std::vector<std::vector<std::vector<size_t>>>& table) {
  for(uint64_t iter=0; iter<table.size(); ++iter) {
    std::cout << "~~~~~~~~~~~~~ Iter: " << iter << " ~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    for(uint64_t char_ind=0; char_ind<table[iter].size(); ++char_ind) {
      std::cout << "num chars: " << table[iter][char_ind].size() << " - ";
      for(uint64_t needle_ind=0; needle_ind<table[iter][char_ind].size(); ++needle_ind) {
        std::cout << needles[table[iter][char_ind][needle_ind]][char_ind] << ", ";
      }
      std::cout << std::endl;
    }
    std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  }
}

void string_match(std::vector<std::string>& needles, std::string& haystack, std::vector<std::vector<std::vector<size_t>>>& needles_table, bool is_horizontal, std::vector<int>& matches) {
  pimStartTimer();
  // TODO update types when pim type conversion operation is available, currently everything uses PIM_UINT32, however this is unecessary

  // If vertical, each pim object takes 32 rows, 1 row if horizontal
  // Two rows used by the haystack and intermediate

  PimObjId haystack_pim = pimAlloc(PIM_ALLOC_AUTO, haystack.size(), PIM_UINT32);
  assert(haystack_pim != -1);
  
  PimObjId intermediate_pim = pimAllocAssociated(haystack_pim, PIM_UINT32);
  assert(intermediate_pim != -1);

  // TODO remove when pim conversion becomes possible
  uint32_t *haystack_32bit = (uint32_t*) malloc(sizeof(uint32_t) * haystack.size());
  #pragma omp parallel for
  for(uint32_t i=0; i<haystack.size(); ++i) {
    haystack_32bit[i] = (uint32_t) haystack[i];
  }

  std::vector<PimObjId> pim_individual_needle_matches;
  for(uint32_t i=0; i<needles_table[0][0].size(); ++i) {
    pim_individual_needle_matches.push_back(pimAllocAssociated(haystack_pim, PIM_UINT32));
    assert(pim_individual_needle_matches.back() != -1);
  }

  uint64_t needles_done = 0;

  PimStatus status;
  for(uint64_t iter=0; iter<needles_table.size(); ++iter) {
    // TODO: would it be better to save a copy of the haystack on the device then copy it back from the device?
    status = pimCopyHostToDevice((void *)haystack_32bit, haystack_pim);
    assert (status == PIM_OK);

    uint64_t first_avail_pim_needle_result = iter == 0 ? 0 : 1;

    for(uint64_t char_idx=0; char_idx < needles_table[iter].size(); ++char_idx) {

      char prev_char = '\0';
      
      for(uint64_t needle_idx=0; needle_idx < needles_table[iter][char_idx].size(); ++needle_idx) {
        
        uint64_t current_needle_idx = needles_table[iter][char_idx][needle_idx];
        uint64_t needle_idx_pim = (current_needle_idx - needles_done) + first_avail_pim_needle_result;
        char current_char = needles[current_needle_idx][char_idx];

        if(char_idx == 0) {
          status = pimEQScalar(haystack_pim, pim_individual_needle_matches[needle_idx_pim], (uint64_t) current_char);
          assert (status == PIM_OK);
        } else if(prev_char == current_char) {
          status = pimAnd(pim_individual_needle_matches[needle_idx_pim], intermediate_pim, pim_individual_needle_matches[needle_idx_pim]);
          assert (status == PIM_OK);
        } else {
          status = pimEQScalar(haystack_pim, intermediate_pim, (uint64_t) current_char);
          assert (status == PIM_OK);

          status = pimAnd(pim_individual_needle_matches[needle_idx_pim], intermediate_pim, pim_individual_needle_matches[needle_idx_pim]);
          assert (status == PIM_OK);
        }
        prev_char = current_char;
      }

      if(char_idx + 1 < needles_table[iter].size()) {
        status = pimShiftElementsLeft(haystack_pim);
        assert (status == PIM_OK);
      }
    }

    for(uint64_t needle_idx = 0; needle_idx < needles_table[iter][0].size(); ++needle_idx) {
      uint64_t current_needle_idx = needle_idx + needles_done;
      uint64_t needle_idx_pim = (current_needle_idx - needles_done) + first_avail_pim_needle_result;

      if(is_horizontal) {
        status = pimMulScalar(pim_individual_needle_matches[needle_idx_pim], pim_individual_needle_matches[needle_idx_pim], 1 + current_needle_idx);
        assert (status == PIM_OK);
      } else {
        status = pimXorScalar(pim_individual_needle_matches[needle_idx_pim], pim_individual_needle_matches[needle_idx_pim], 1);
        assert (status == PIM_OK);

        status = pimSubScalar(pim_individual_needle_matches[needle_idx_pim], pim_individual_needle_matches[needle_idx_pim], 1);
        assert (status == PIM_OK);

        status = pimAndScalar(pim_individual_needle_matches[needle_idx_pim], pim_individual_needle_matches[needle_idx_pim], 1 + current_needle_idx);
        assert (status == PIM_OK);
      }
    }

    for(uint64_t needle_idx = 1; needle_idx < needles_table[iter][0].size() + first_avail_pim_needle_result; ++needle_idx) {
      status = pimMax(pim_individual_needle_matches[0], pim_individual_needle_matches[needle_idx], pim_individual_needle_matches[0]);
      assert (status == PIM_OK);
    }

    needles_done += needles_table[iter][0].size();
  }

  status = pimCopyDeviceToHost(pim_individual_needle_matches[0], (void *)matches.data());
  assert (status == PIM_OK);
  pimEndTimer();
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
  
  if (!createDevice(params.configFile))
  {
    return 1;
  }

  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  assert(status == PIM_OK);

  matches.resize(haystack.size(), 0);

  std::vector<std::vector<std::vector<size_t>>> table = string_match_precompute_table(needles, 2 * deviceProp.numRowPerSubarray, deviceProp.isHLayoutDevice);
  // print_table(needles, table);
  
  string_match(needles, haystack, table, deviceProp.isHLayoutDevice, matches);

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
