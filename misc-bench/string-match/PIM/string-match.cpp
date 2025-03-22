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
#include "utilStringMatch.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  const char *keysInputFile;
  const char *textInputFile;
  char *configFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./string-match.out [options]"
          "\n"
          "\n    -k    keys input file, each key on new line (default=dataset/10mil_l-10_nk-10_kl/keys.txt) must be sorted by increasing length, must have a blank line at end of file"
          "\n    -t    text input file to search for keys from (default=dataset/10mil_l-10_nk-10_kl/text.txt)"
          "\n    -c    dramsim config file"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.keysInputFile = "./../dataset/10mil_l-10_nk-10_kl/keys.txt";
  p.textInputFile = "./../dataset/10mil_l-10_nk-10_kl/text.txt";
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
// Instead of calling pimEQScalar multiple times for the same character, call only once per character and reuse result
// Not included in benchmarking time, as it only depends on the keys/needles and not on the text/haystack
std::vector<std::vector<std::vector<size_t>>> stringMatchPrecomputeTable(std::vector<std::string>& needles, uint64_t numRows, bool isHorizontal) {
  std::vector<std::vector<std::vector<size_t>>> resultTable;
  
  // Maximize the number of needles computed each iteration
  constexpr uint64_t verticalNonNeedleRows = 8 + 1 + 32 + 32; // haystackPim, intermediatePimBool, preResultPim, resultPim
  constexpr uint64_t horizontalNonNeedleRows = 4; // haystackPim, intermediatePimBool, preResultPim, resultPim
  uint64_t maxNeedlesPerIterationOneIteration = numRows - (isHorizontal ? horizontalNonNeedleRows : verticalNonNeedleRows);
  constexpr uint64_t verticalHaystackCopyRows = 8; // haystackCopyPim
  constexpr uint64_t horizontalHaystackCopyRows = 1; // haystackCopyPim
  uint64_t maxNeedlesPerIterationMultipleIterations = maxNeedlesPerIterationOneIteration - (isHorizontal ? horizontalHaystackCopyRows : verticalHaystackCopyRows); // Require space for haystack copy if more than one iteration
  uint64_t numIterations;

  if(needles.size() <= maxNeedlesPerIterationOneIteration) {
    numIterations = 1;
  } else {
    numIterations = (needles.size() + maxNeedlesPerIterationMultipleIterations - 1) / maxNeedlesPerIterationMultipleIterations;
  }

  resultTable.resize(numIterations);

  uint64_t needlesDone = 0;

  for(uint64_t iter=0; iter<numIterations; ++iter) {
    uint64_t needlesThisIteration;
    if(numIterations == 1) {
      needlesThisIteration = needles.size();
    } else if (iter+1 < numIterations){
      needlesThisIteration = maxNeedlesPerIterationMultipleIterations;
    } else {
      needlesThisIteration = needles.size() - needlesDone;
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

      // Sorting places identical characters next to each other, so their equality results can be reused
      std::sort(currentTableRow.begin(), currentTableRow.end(), [&needles, &charInd](auto& l, auto& r) {
        return needles[l][charInd] < needles[r][charInd];
      });
    }

    needlesDone += needlesThisIteration;
  }

  return resultTable;
}

void stringMatch(std::vector<std::string>& needles, std::string& haystack, std::vector<std::vector<std::vector<size_t>>>& needlesTable, bool isHorizontal, std::vector<int>& matches) {
  PimStatus status;
  
  PimObjId resultPim = pimAlloc(PIM_ALLOC_AUTO, haystack.size(), PIM_UINT32);
  assert(resultPim != -1);

  // Stores the text that is being checked for the needles
  PimObjId haystackPim = pimAllocAssociated(resultPim, PIM_UINT8);
  assert(haystackPim != -1);
  
  // Used for intermediate calculations
  PimObjId intermediatePimBool = pimAllocAssociated(resultPim, PIM_BOOL);
  assert(intermediatePimBool != -1);

  PimObjId preResultPim = pimAllocAssociated(resultPim, PIM_UINT32);
  assert(preResultPim != -1);

  status = pimCopyHostToDevice((void *)haystack.data(), haystackPim);
  assert (status == PIM_OK);

  PimObjId haystackCopyPim = -1;
  if(needlesTable.size() > 1) {
    haystackCopyPim = pimAllocAssociated(haystackPim, PIM_UINT8);
    assert(haystackCopyPim != -1);

    status = pimCopyObjectToObject(haystackPim, haystackCopyPim);
    assert (status == PIM_OK);
  }
  

  if(needlesTable.empty() || needlesTable[0].empty()) {
    std::cerr << "Error: The needles table is empty" << std::endl;
    exit(1);
  }

  // Matches are calculated for a group of needles at a time, this vector stores the matches for each needle
  // needlesTable[0][0].size() is the number of needles in the first iteration, which will have at least the most needles out of all iterations
  size_t maxNeedlesInOneIteration = needlesTable[0][0].size();

  std::vector<PimObjId> pimIndividualNeedleMatchesBool(maxNeedlesInOneIteration);
  for (size_t i = 0; i < maxNeedlesInOneIteration; ++i) {
    pimIndividualNeedleMatchesBool[i] = pimAllocAssociated(resultPim, PIM_BOOL);
    assert(pimIndividualNeedleMatchesBool[i] != -1);
  }

  uint64_t needlesDone = 0;

  pimStartTimer();

  // Number of needles at a time is limited by how many PIM objects can be alloc associated with each other in a subarray
  // Iterates multiple times if there are enough needles
  for(uint64_t iter=0; iter<needlesTable.size(); ++iter) {

    // Iterate through each character index, as determined in the precomputing step
    // e.g.: needles = ["abc", "def"]
    // first iteration will process 'a' and 'd'
    // this allows amortization of the pimShiftElementsLeft calls
    for(uint64_t charIdx=0; charIdx < needlesTable[iter].size(); ++charIdx) {

      // Stores the last character checked using pimEQScalar
      char prevChar = '\0';
      
      // Iterates through the needles, checking the characters at the index charIdx
      // Processes in an order for optimal reuse
      // e.g.: needles = ["abc", "dec", "aab"]
      // will be processed in the following order:
      // charIdx==0: 'a', 'a', 'd' -> can reuse pimEQScalar call for the two 'a's
      // charIdx==1: 'a', 'b', 'e'
      // charIdx==2: 'b', 'c', 'c' -> can reuse pimEQScalar call for the two 'c's
      // This optimization is helpful whenver there are multiple of the same character in multiple needles at the same index
      // This is more likely to happen when there are a lot of needles/keys

      for(uint64_t needleIdx=0; needleIdx < needlesTable[iter][charIdx].size(); ++needleIdx) {
        
        uint64_t needleIdxHost = needlesTable[iter][charIdx][needleIdx]; // Can be used to index into needles
        uint64_t needleIdxPim = needleIdxHost - needlesDone; // Can be used to index into pimIndividualNeedleMatches
        char currentChar = needles[needleIdxHost][charIdx];

        if(charIdx == 0) {
          // If on the first character index, there is no need to pimAnd with the current possible matches
          // Instead, place the equality result directly into the match array
          status = pimEQScalar(haystackPim, pimIndividualNeedleMatchesBool[needleIdxPim], (uint64_t) currentChar);
          assert (status == PIM_OK);
        } else if(prevChar == currentChar) {
          // Reuse the previously calculated equality result in intermediatePim and pimAnd with the current matches
          status = pimAnd(pimIndividualNeedleMatchesBool[needleIdxPim], intermediatePimBool, pimIndividualNeedleMatchesBool[needleIdxPim]);
          assert (status == PIM_OK);
        } else {
          // Check the entirety of the text if it is equal with the current character
          status = pimEQScalar(haystackPim, intermediatePimBool, (uint64_t) currentChar);
          assert (status == PIM_OK);

          // Update the potential match array
          status = pimAnd(pimIndividualNeedleMatchesBool[needleIdxPim], intermediatePimBool, pimIndividualNeedleMatchesBool[needleIdxPim]);
          assert (status == PIM_OK);
        }
        prevChar = currentChar;
      }

      // Shift the haystack to the left to check the next character in it
      // Only shift if there will be another round of character checks in this iteration
      if(charIdx + 1 < needlesTable[iter].size()) {
        status = pimShiftElementsLeft(haystackPim);
        assert (status == PIM_OK);
      }
    }

    for(uint64_t needleIdx = 0; needleIdx < needlesTable[iter][0].size(); ++needleIdx) {
      uint64_t needleIdxHost = needleIdx + needlesDone; // Can be used to index into needles
      uint64_t needleIdxPim = needleIdx; // Can be used to index into pimIndividualNeedleMatches

      // Switch to conditional operations
      status = pimBroadcastUInt(preResultPim, 0);
      assert (status == PIM_OK);
      status = pimCondBroadcast(pimIndividualNeedleMatchesBool[needleIdxPim], 1 + needleIdxHost, preResultPim);
      assert (status == PIM_OK);

      // Update the final result array with the matches from this iteration
      // In the problem statement, we specify that the longest needle should be given as a result if multiple match at the same position
      // needleIdxHost will be larger if the needle is longer, because we specify that needles should be sorted ahead of time
      // Therefore, to get the longest needle at each position, we do a max reduction on the pimIndividualNeedleMatches[needleIdx] objects
      status = pimMax(resultPim, preResultPim, resultPim);
      assert (status == PIM_OK);
    }

    needlesDone += needlesTable[iter][0].size();

    if(iter + 1 < needlesTable.size()) {
      status = pimCopyObjectToObject(haystackCopyPim, haystackPim);
      assert (status == PIM_OK);
    }
  }

  status = pimCopyDeviceToHost(resultPim, (void *)matches.data());
  assert (status == PIM_OK);

  pimEndTimer();

  status = pimFree(haystackPim);
  assert (status == PIM_OK);

  status = pimFree(intermediatePimBool);
  assert (status == PIM_OK);

  for(PimObjId individualNeedleMatchBool : pimIndividualNeedleMatchesBool) {
    status = pimFree(individualNeedleMatchBool);
    assert (status == PIM_OK);
  }
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  
  std::cout << "Running PIM string match for \"" << params.keysInputFile << "\" as the keys file, and \"" << params.textInputFile << "\" as the text input file\n";
  
  std::string haystack;
  std::vector<std::string> needles;
  std::vector<int> matches;

  haystack = readStringFromFile(params.textInputFile);
  needles = getNeedlesFromFile(params.keysInputFile);
  
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
        std::cerr << "Wrong answer: " << unsigned(matches[i]) << " (expected " << unsigned(matchesCpu[i]) << "), for position " << i << std::endl;
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
