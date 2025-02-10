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

#include "util.h"
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

// Precomputes optimal order to match keys/needles for calculation reuse
// Not included in benchmarking time, as it only depends on the keys/needles and not on the text/haystack
std::vector<std::vector<std::vector<size_t>>> stringMatchPrecomputeTable(std::vector<std::string>& needles, uint64_t numRows, bool isHorizontal) {
  std::vector<std::vector<std::vector<size_t>>> resultTable;
  
  uint64_t maxNeedlesPerIteration = isHorizontal ? numRows - 2 : (numRows>>5) - 2;
  uint64_t numIterations;
  if(needles.size() <= maxNeedlesPerIteration) {
    numIterations = 1;
  } else {
    uint64_t needlesAfterFirstIteration = needles.size() - maxNeedlesPerIteration;
    // Can do 1 more needle in first iteration than in later iterations
    // During the first iteration we can use the final result array as an individual result array in the first iteration
    numIterations = 1 + ((needlesAfterFirstIteration + maxNeedlesPerIteration - 2) / (maxNeedlesPerIteration - 1));
  }

  resultTable.resize(numIterations);

  uint64_t needlesDone = 0;

  for(uint64_t iter=0; iter<numIterations; ++iter) {
    uint64_t needlesThisIteration;
    if(iter == 0) {
      needlesThisIteration = min(maxNeedlesPerIteration, needles.size());
    } else if(iter+1 == numIterations) {
      needlesThisIteration = needles.size() - needlesDone;
    } else {
      needlesThisIteration = maxNeedlesPerIteration - 1;
    }

    // Range: [needlesDone, needlesDone + needlesThisIteration - 1]
    uint64_t firstNeedleThisIteration = needlesDone;
    uint64_t lastNeedleThisIteration = firstNeedleThisIteration + needlesThisIteration - 1;
    uint64_t longestNeedleThisIteration = needles[lastNeedleThisIteration].size();
    // As we iterate through character indices for the needles in this iteration, there may be some needles that are shorter than the current character
    // Skip checking them by keeping track of the shortest needle that is long enough to have the current character
    uint64_t currentStartNeedle = firstNeedleThisIteration;
    resultTable[iter].resize(longestNeedleThisIteration);
    for(uint64_t charInd = 0; charInd < longestNeedleThisIteration; ++charInd) {
      while(needles[currentStartNeedle].size() <= charInd) {
        ++currentStartNeedle;
      }
      std::vector<size_t>& currentTableRow = resultTable[iter][charInd];
      currentTableRow.resize(1 + lastNeedleThisIteration - currentStartNeedle);
      // Sort needles [currentStartNeedle, lastNeedleThisIteration] on charInd
      
      std::iota(currentTableRow.begin(), currentTableRow.end(), currentStartNeedle);
      std::sort(currentTableRow.begin(), currentTableRow.end(), [&needles, &charInd](auto& l, auto& r) {
        return needles[l][charInd] < needles[r][charInd];
      });
    }

    needlesDone += needlesThisIteration;
  }

  return resultTable;
}

void stringMatch(std::vector<std::string>& needles, std::string& haystack, std::vector<std::vector<std::vector<size_t>>>& needlesTable, bool isHorizontal, std::vector<int>& matches) {
  // TODO update types when pim type conversion operation is available, currently everything uses PIM_UINT32, however this is unecessary

  // If vertical, each pim object takes 32 rows, 1 row if horizontal
  // Two rows used by the haystack and intermediate

  PimObjId haystackPim = pimAlloc(PIM_ALLOC_AUTO, haystack.size(), PIM_UINT32);
  assert(haystackPim != -1);
  
  PimObjId intermediatePim = pimAllocAssociated(haystackPim, PIM_UINT32);
  assert(intermediatePim != -1);

  // TODO remove when pim conversion becomes possible
  uint32_t *haystack32Bit = (uint32_t*) malloc(sizeof(uint32_t) * haystack.size());
  #pragma omp parallel for
  for(uint32_t i=0; i<haystack.size(); ++i) {
    haystack32Bit[i] = (uint32_t) haystack[i];
  }

  std::vector<PimObjId> pimIndividualNeedleMatches;
  for(uint32_t i=0; i<needlesTable[0][0].size(); ++i) {
    pimIndividualNeedleMatches.push_back(pimAllocAssociated(haystackPim, PIM_UINT32));
    assert(pimIndividualNeedleMatches.back() != -1);
  }

  uint64_t needlesDone = 0;

  pimStartTimer();

  PimStatus status;
  for(uint64_t iter=0; iter<needlesTable.size(); ++iter) {
    status = pimCopyHostToDevice((void *)haystack32Bit, haystackPim);
    assert (status == PIM_OK);

    uint64_t firstAvailPimNeedleResult = iter == 0 ? 0 : 1;

    for(uint64_t charIdx=0; charIdx < needlesTable[iter].size(); ++charIdx) {

      char prevChar = '\0';
      
      for(uint64_t needleIdx=0; needleIdx < needlesTable[iter][charIdx].size(); ++needleIdx) {
        
        uint64_t currentNeedleIdx = needlesTable[iter][charIdx][needleIdx];
        uint64_t needleIdxPim = (currentNeedleIdx - needlesDone) + firstAvailPimNeedleResult;
        char currentChar = needles[currentNeedleIdx][charIdx];

        if(charIdx == 0) {
          status = pimEQScalar(haystackPim, pimIndividualNeedleMatches[needleIdxPim], (uint64_t) currentChar);
          assert (status == PIM_OK);
        } else if(prevChar == currentChar) {
          status = pimAnd(pimIndividualNeedleMatches[needleIdxPim], intermediatePim, pimIndividualNeedleMatches[needleIdxPim]);
          assert (status == PIM_OK);
        } else {
          status = pimEQScalar(haystackPim, intermediatePim, (uint64_t) currentChar);
          assert (status == PIM_OK);

          status = pimAnd(pimIndividualNeedleMatches[needleIdxPim], intermediatePim, pimIndividualNeedleMatches[needleIdxPim]);
          assert (status == PIM_OK);
        }
        prevChar = currentChar;
      }

      if(charIdx + 1 < needlesTable[iter].size()) {
        status = pimShiftElementsLeft(haystackPim);
        assert (status == PIM_OK);
      }
    }

    for(uint64_t needleIdx = 0; needleIdx < needlesTable[iter][0].size(); ++needleIdx) {
      uint64_t currentNeedleIdx = needleIdx + needlesDone;
      uint64_t needleIdxPim = (currentNeedleIdx - needlesDone) + firstAvailPimNeedleResult;

      if(isHorizontal) {
        status = pimMulScalar(pimIndividualNeedleMatches[needleIdxPim], pimIndividualNeedleMatches[needleIdxPim], 1 + currentNeedleIdx);
        assert (status == PIM_OK);
      } else {
        status = pimXorScalar(pimIndividualNeedleMatches[needleIdxPim], pimIndividualNeedleMatches[needleIdxPim], 1);
        assert (status == PIM_OK);

        status = pimSubScalar(pimIndividualNeedleMatches[needleIdxPim], pimIndividualNeedleMatches[needleIdxPim], 1);
        assert (status == PIM_OK);

        status = pimAndScalar(pimIndividualNeedleMatches[needleIdxPim], pimIndividualNeedleMatches[needleIdxPim], 1 + currentNeedleIdx);
        assert (status == PIM_OK);
      }
    }

    for(uint64_t needleIdx = 1; needleIdx < needlesTable[iter][0].size() + firstAvailPimNeedleResult; ++needleIdx) {
      status = pimMax(pimIndividualNeedleMatches[0], pimIndividualNeedleMatches[needleIdx], pimIndividualNeedleMatches[0]);
      assert (status == PIM_OK);
    }

    needlesDone += needlesTable[iter][0].size();
  }

  status = pimCopyDeviceToHost(pimIndividualNeedleMatches[0], (void *)matches.data());
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
  
  if (!createDevice(params.configFile))
  {
    return 1;
  }

  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  assert(status == PIM_OK);

  matches.resize(haystack.size(), 0);

  std::vector<std::vector<std::vector<size_t>>> table = stringMatchPrecomputeTable(needles, 2 * deviceProp.numRowPerSubarray, deviceProp.isHLayoutDevice);
  
  stringMatch(needles, haystack, table, deviceProp.isHLayoutDevice, matches);

  if (params.shouldVerify) 
  {
    std::vector<int> matchesCpu;
    
    matchesCpu.resize(haystack.size());

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

  pimShowStats();

  return 0;
}
