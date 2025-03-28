// Test: C++ version of hamming string match
// Copyright (c) 2025 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

// The output format is based on PFAC by Lin et. al.
// This version of string matching only matches one key per position in the text.
// However, it could be modified to any of the below formats with few changes.
// Example: text="abcda", keys=["a", "abc"], maxHammingDistance=0
// PFAC Style Output (Current): Array of the length of the text, where each position is a 0 (no match), or the index of the key that matches (starts at 1).
//      Example Output: [2, 0, 0, 0, 1].
//      Note that for this format, each position can only have one matching key.
//      For the regular matching, the longer key takes priority. For Hamming matching, the priorities are set arbitrarily in advance.
// Binary match array for each key:
//      Example Output: [[1, 0, 0, 0, 1], [1, 0, 0, 0, 0]]
//      Would be done by copying the binary match arrays back to the host.
// Number of matches for each key:
//      Example Output: [2, 1]
//      Would be done by doing a reduction sum on each binary match array.
// Binary match array for all keys (phoenix-like):
//      Example Output: [1, 1]
//      Would be done using a max reduction on each binary match array.

// Format that is not reasonably implementable in PIM:
// Matching positions for each key:
//      Example Output: [[0, 4], [0]]

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <numeric>
#include <stdexcept>
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
  const char *hammingDistanceInputFile;
  char *configFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./hamming-string-match.out [options]"
          "\n"
          "\n    -k    keys input file, each key on new line (default=dataset/10mil_l-10_nk-10_kl/keys.txt), must have a blank line at end of file"
          "\n    -t    text input file to search for keys from (default=dataset/10mil_l-10_nk-10_kl/text.txt)"
          "\n    -d    max hamming distance file (default=dataset/10mil_l-10_nk-10_kl/maxHammingDistance.txt)"
          "\n    -c    dramsim config file"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.keysInputFile = "./../dataset/10mil_l-10_nk-10_kl/keys.txt";
  p.textInputFile = "./../dataset/10mil_l-10_nk-10_kl/text.txt";
  p.hammingDistanceInputFile = "./../dataset/10mil_l-10_nk-10_kl/maxHammingDistance.txt";
  p.configFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:k:t:d:c:v:")) >= 0)
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
    case 'd':
      p.hammingDistanceInputFile = optarg;
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

//! @brief Precomputed list of the needles to match and the order to process the characters
struct NeedlesTable {
  using NeedlesTableList = std::vector<std::vector<std::vector<size_t>>>;

  //! @brief Defines the order in which to process the characters of all needles
  NeedlesTableList needlesCharsInOrder;

  //! @brief Specifies the needles that are ending (out of characters) for each step
  NeedlesTableList needlesEnding;

  //! @brief Says how to get from actual needle index to sorted needle index
  std::vector<size_t> actualToSortedNeedles;
};

//! @brief  Precomputes a more optimal order to match keys/needles on PIM device for calculation reuse
//! @details Instead of calling pimNEScalar multiple times for the same character, call only once per character per char index and reuse result
//!          Orders needles within a character index to place needles with the same character next to each other
//!          Not included in benchmarking time as it is a one time cost that can be reused to match against multiple haystacks
//! @param[in]  needles  A list of needles that will be matched
//! @param[in]  numRows  The number of rows in the PIM device, needed to find the max number of needles per iteration
//! @param[in]  isHorizontal  Represents if the PIM device is horizontal, needed to find the max number of needles per iteration
//! @return  A table representing a better ordering to match the needles
NeedlesTable stringMatchPrecomputeTable(const std::vector<std::string>& needles, const uint64_t numRows, const bool isHorizontal) {
  using NeedlesTableList = std::vector<std::vector<std::vector<size_t>>>;
  NeedlesTable resultTable;
  NeedlesTableList& resultTableOrdering = resultTable.needlesCharsInOrder;
  NeedlesTableList& resultTableEnding = resultTable.needlesEnding;
  std::vector<size_t>& resultTableActualToSorted = resultTable.actualToSortedNeedles;

  std::vector<size_t> sortedToActualNeedles(needles.size());
  std::iota(sortedToActualNeedles.begin(), sortedToActualNeedles.end(), 0);
  std::sort(sortedToActualNeedles.begin(), sortedToActualNeedles.end(), [&needles](auto l, auto r) {
    return needles[l].size() < needles[r].size();
  });

  resultTableActualToSorted.resize(needles.size());
  for(uint64_t needleIdx=0; needleIdx<needles.size(); ++needleIdx) {
    resultTableActualToSorted[sortedToActualNeedles[needleIdx]] = needleIdx;
  }

  // Maximize the number of needles computed each iteration
  constexpr uint64_t verticalNonNeedleRows = 8 + 1 + 1; // haystackPim, intermediatePimBool, isHaystackNonZeroPimBool
  constexpr uint64_t horizontalNonNeedleRows = 3; // haystackPim, intermediatePimBool, isHaystackNonZeroPimBool
  constexpr uint64_t verticalHaystackCopyRows = 8; // haystackCopyPim
  constexpr uint64_t horizontalHaystackCopyRows = 1; // haystackCopyPim
  constexpr uint64_t verticalRowsPerNeedle = 32;
  constexpr uint64_t horizontalRowsPerNeedle = 1;

  const uint64_t needleRowsOneIteration = numRows - (isHorizontal ? horizontalNonNeedleRows : verticalNonNeedleRows);
  // Require space for haystack copy if running for more than one iteration
  const uint64_t needleRowsMultipleIterations = needleRowsOneIteration - (isHorizontal ? horizontalHaystackCopyRows : verticalHaystackCopyRows);
  const uint64_t rowsPerNeedle = isHorizontal ? horizontalRowsPerNeedle : verticalRowsPerNeedle;
  const uint64_t maxNeedlesPerIterationOneIteration = needleRowsOneIteration / rowsPerNeedle;
  const uint64_t maxNeedlesPerIterationMultipleIterations = needleRowsMultipleIterations / rowsPerNeedle;

  uint64_t numIterations;
  if(needles.size() <= maxNeedlesPerIterationOneIteration) {
    numIterations = 1;
  } else {
    uint64_t needlesAfterFirstIteration = needles.size() - maxNeedlesPerIterationMultipleIterations;
    // Can do 1 more needle in first iteration than in later iterations
    // During the first iteration we can use the final result array as an individual result array
    numIterations = 1 + ((needlesAfterFirstIteration + maxNeedlesPerIterationOneIteration - 2) / (maxNeedlesPerIterationOneIteration - 1));
  }

  resultTableOrdering.resize(numIterations);
  resultTableEnding.resize(numIterations);

  uint64_t needlesDone = 0;

  for(uint64_t iter=0; iter<numIterations; ++iter) {
    uint64_t needlesThisIteration;
    if(numIterations == 1) {
      needlesThisIteration = needles.size();
    } else if(iter == 0) {
      needlesThisIteration = maxNeedlesPerIterationMultipleIterations;
    } else if(iter+1 == numIterations) {
      needlesThisIteration = needles.size() - needlesDone;
    } else {
      needlesThisIteration = maxNeedlesPerIterationMultipleIterations - 1;
    }

    // Range: [needlesDone, needlesDone + needlesThisIteration - 1]
    uint64_t firstNeedleThisIteration = needlesDone;
    uint64_t lastNeedleThisIteration = firstNeedleThisIteration + needlesThisIteration - 1;
    uint64_t longestNeedleThisIteration = needles[sortedToActualNeedles[lastNeedleThisIteration]].size();
    // As we iterate through character indices for the needles in this iteration, there may be some needles that are shorter than the current character
    // Skip checking them by keeping track of the shortest needle that is long enough to have the current character
    uint64_t currentStartNeedle = firstNeedleThisIteration;
    resultTableOrdering[iter].resize(longestNeedleThisIteration);
    for(uint64_t charInd = 0; charInd < longestNeedleThisIteration; ++charInd) {
      while(needles[sortedToActualNeedles[currentStartNeedle]].size() <= charInd) {
        ++currentStartNeedle;
      }
      std::vector<size_t>& currentTableRow = resultTableOrdering[iter][charInd];
      currentTableRow.resize(1 + lastNeedleThisIteration - currentStartNeedle);
      // Sort needles [currentStartNeedle, lastNeedleThisIteration] on charInd
      
      for(uint64_t currentTableRowIdx=0; currentTableRowIdx<currentTableRow.size(); ++currentTableRowIdx) {
        currentTableRow[currentTableRowIdx] = sortedToActualNeedles[currentStartNeedle + currentTableRowIdx];
      }

      // Sorting places identical characters next to each other, so their equality results can be reused
      std::sort(currentTableRow.begin(), currentTableRow.end(), [&needles, &charInd](auto& l, auto& r) {
        return needles[l][charInd] < needles[r][charInd];
      });
    }

    // Stores all needles that are ending at this step
    resultTableEnding[iter].resize(longestNeedleThisIteration);
    for(uint64_t needleIdx=firstNeedleThisIteration; needleIdx <= lastNeedleThisIteration; ++needleIdx) {
      uint64_t actualNeedleIdx = sortedToActualNeedles[needleIdx];
      resultTableEnding[iter][needles[actualNeedleIdx].size() - 1].push_back(actualNeedleIdx);
    }

    needlesDone += needlesThisIteration;
  }

  return resultTable;
}

//! @brief  Matches strings on a PIM device with an allowed number of character differences
//! @param[in]  needles  A list of strings to search for
//! @param[in]  haystack  A string to search within
//! @param[in]  maxHammingDistance  The maximum possible number of characters that can be different for finding a match
//! @param[in]  needlesTable  A pre generated table that defines an efficient way to process the needles, created by stringMatchPrecomputeTable()
//! @param[in]  isHorizontal  Specifies if the PIM device is horizontal or vertical
//! @param[out] matches The result, a list of matches of the size of the haystack
void hammingStringMatch(const std::vector<std::string>& needles, const std::string& haystack, const uint64_t maxHammingDistance, const NeedlesTable& needlesTable, const bool isHorizontal, std::vector<int>& matches) {
  using NeedlesTableList = std::vector<std::vector<std::vector<size_t>>>;
  const NeedlesTableList& needlesTableOrdering = needlesTable.needlesCharsInOrder;
  const NeedlesTableList& needlesTableEnding = needlesTable.needlesEnding;
  const std::vector<size_t>& needlesTableActualToSorted = needlesTable.actualToSortedNeedles;
  
  PimStatus status;

  if(needlesTableOrdering.empty() || needlesTableOrdering[0].empty() || needlesTableEnding.empty() || needlesTableEnding[0].empty()) {
    std::cerr << "Error: The needles table is empty" << std::endl;
    exit(1);
  }

  // needlesTableOrdering[0][0].size() is the number of needles in the first iteration, which will have the most needles out of all iterations
  size_t maxNeedlesInOneIteration = needlesTableOrdering[0][0].size();

  // Matches are calculated for a group of needles at a time, this vector stores the mismatches for each needle
  // Each pimIndividualNeedleMismatches[i] contains an array of the number of mismatches for a given needle at each position in the haystack
  std::vector<PimObjId> pimIndividualNeedleMismatches(maxNeedlesInOneIteration);
  pimIndividualNeedleMismatches[0] = pimAlloc(PIM_ALLOC_AUTO, haystack.size(), PIM_UINT32);
  assert(pimIndividualNeedleMismatches[0] != -1);
  PimObjId rootObjToAssoc = pimIndividualNeedleMismatches[0];
  for(size_t i=1; i<maxNeedlesInOneIteration; ++i) {
    pimIndividualNeedleMismatches[i] = pimAllocAssociated(rootObjToAssoc, PIM_UINT32);
    assert(pimIndividualNeedleMismatches[i] != -1);
  }

  // Stores the text that is being checked for the needles
  PimObjId haystackPim = pimAllocAssociated(rootObjToAssoc, PIM_UINT8);
  assert(haystackPim != -1);
  status = pimCopyHostToDevice((void *) haystack.data(), haystackPim);
  assert (status == PIM_OK);
  
  // Used for intermediate calculations
  PimObjId intermediatePimBool = pimAllocAssociated(rootObjToAssoc, PIM_BOOL);
  assert(intermediatePimBool != -1);

  // Used for checking if haystack elements are non-zero
  PimObjId isHaystackNonZeroPimBool = pimAllocAssociated(rootObjToAssoc, PIM_BOOL);
  assert(isHaystackNonZeroPimBool != -1);

  // Haystack copy is only necessary if there is more than one iteration
  PimObjId haystackCopyPim = -1;
  if(needlesTableOrdering.size() > 1) {
    haystackCopyPim = pimAllocAssociated(rootObjToAssoc, PIM_UINT8);
    assert(haystackCopyPim != -1);

    status = pimCopyObjectToObject(haystackPim, haystackCopyPim);
    assert (status == PIM_OK);
  }

  uint64_t needlesDone = 0;

  pimStartTimer();

  // Number of needles at a time is limited by how many PIM objects can be alloc associated with each other in a subarray
  // Iterates multiple times if there are enough needles
  for(uint64_t iter=0; iter<needlesTableOrdering.size(); ++iter) {
    // Instead of creating a seperate result variable, we reuse one of the objects in pimIndividualNeedleMismatches
    // Can be used as a needle match array for only the first iteration, and a result array for the rest
    // Slight optimization allows one extra needle in the first iteration
    uint64_t firstAvailPimNeedleResult = iter == 0 ? 0 : 1;

    // Iterate through each character index, as determined in the precomputing step
    // e.g.: needles = ["abc", "def"]
    // first iteration will process 'a' and 'd'
    // this allows amortization of the pimShiftElementsLeft calls
    for(uint64_t charIdx=0; charIdx < needlesTableOrdering[iter].size(); ++charIdx) {

      // Stores the last character checked using pimNEScalar
      char prevChar = '\0';
      
      // Iterates through the needles, checking the characters at the index charIdx
      // Processes in an order for optimal reuse
      // e.g.: needles = ["abc", "dec", "aab"]
      // will be processed in the following order:
      // charIdx==0: 'a', 'a', 'd' -> can reuse pimNEScalar result for the two 'a's
      // charIdx==1: 'a', 'b', 'e' -> cannot reuse pimNEScalar result
      // charIdx==2: 'b', 'c', 'c' -> can reuse pimNEScalar result for the two 'c's
      // This optimization is helpful whenver there are multiple copies of the same character in multiple needles at the same index
      // This is more likely to happen when there are a lot of needles/keys

      for(uint64_t needleIdx=0; needleIdx < needlesTableOrdering[iter][charIdx].size(); ++needleIdx) {
        
        uint64_t needleIdxHost = needlesTableOrdering[iter][charIdx][needleIdx]; // Can be used to index into needles
        
        if(needles[needleIdxHost].size() <= maxHammingDistance) {
          // This means that the string will match everywhere (except for where it would go off the end)
          // Skip matching calculation for this needle for now, will be done later
          continue;
        }
        
        uint64_t needleIdxPim = (needlesTableActualToSorted[needleIdxHost] - needlesDone) + firstAvailPimNeedleResult; // Can be used to index into pimIndividualNeedleMismatches
        char currentChar = needles[needleIdxHost][charIdx];

        if(charIdx == 0) {
          // If on the first character index, there is no need to pimAdd with the current mismatch array
          // Instead, place the equality result directly into the mismatch array
          // Would also work with old comparison, to put result directly into pimIndividualNeedleMismatches[needleIdxPim]
          status = pimNEScalar(haystackPim, intermediatePimBool, (uint64_t) currentChar);
          assert (status == PIM_OK);
          status = pimConvertType(intermediatePimBool, pimIndividualNeedleMismatches[needleIdxPim]);
          assert (status == PIM_OK);
        } else if(prevChar == currentChar) {
          // Reuse the previously calculated equality result in intermediatePimBool and pimAdd with the current matches
          status = pimAdd(pimIndividualNeedleMismatches[needleIdxPim], intermediatePimBool, pimIndividualNeedleMismatches[needleIdxPim]);
          assert (status == PIM_OK);
        } else {
          // Check full text against current character
          status = pimNEScalar(haystackPim, intermediatePimBool, (uint64_t) currentChar);
          assert (status == PIM_OK);

          // Update the mismatch array
          status = pimAdd(pimIndividualNeedleMismatches[needleIdxPim], intermediatePimBool, pimIndividualNeedleMismatches[needleIdxPim]);
          assert (status == PIM_OK);
        }
        prevChar = currentChar;
      }

      if(iter >= needlesTableEnding.size() || charIdx >= needlesTableEnding[iter].size()) {
        std::cerr << "Error: Needles table incorrectly formatted" << std::endl;
        exit(1);
      }

      // This section serves to handle an edge case that can cause matches to go past the edge of the text
      // For example, let haystack="fga", needles=["abc"], and maxHammingDistance=2
      // The array representing the number of mismatches for each position will be [3, 3, 2]
      // This is because the 'a' at the end of the haystack matches the 'a' at the start of the needle
      // Without this fix, the 2 would be less than or equal to the maximum Hamming distance, and it would match at the end
      // However, this is a problem, because the match would be off the edge of the text
      // To solve this, below is an explicit check to elimiate matches that go past the end
      // This is done by checking that the haystack is not 0, as 0's mean that the haystack has been shifted past that position
      // The boolean created in isHaystackNonZeroPimBool below is then anded with the binary match array, ensuring that all matches are within bounds
      if(!needlesTableEnding[iter][charIdx].empty()) {
        status = pimNEScalar(haystackPim, isHaystackNonZeroPimBool, (uint64_t) 0L);
        assert (status == PIM_OK);
      }

      for(uint64_t needleIdx=0; needleIdx < needlesTableEnding[iter][charIdx].size(); ++needleIdx) {

        uint64_t needleIdxHost = needlesTableEnding[iter][charIdx][needleIdx]; // Can be used to index into needles

        if(needles[needleIdxHost].size() <= maxHammingDistance) {
          // This means that the string will match everywhere (except for where it would go off the end)
          // Skip matching calculation for this needle for now, will be done later
          continue;
        }
        
        uint64_t needleIdxPim = (needlesTableActualToSorted[needleIdxHost] - needlesDone) + firstAvailPimNeedleResult; // Can be used to index into pimIndividualNeedleMismatches

        // Checks for matches within maxHammingDistance distance
        // pimIndividualNeedleMismatches[needleIdxPim] represents the hamming distance between the needle and the haystack at each position
        // If pimIndividualNeedleMismatches[needleIdxPim][i] <= maxHammingDistance, there is a match at the position
        status = pimLTScalar(pimIndividualNeedleMismatches[needleIdxPim], intermediatePimBool, maxHammingDistance + 1);
        assert (status == PIM_OK);
        status = pimAnd(intermediatePimBool, isHaystackNonZeroPimBool, intermediatePimBool);
        assert (status == PIM_OK);
        status = pimBroadcastUInt(pimIndividualNeedleMismatches[needleIdxPim], 0);
        assert (status == PIM_OK);
        status = pimCondBroadcast(intermediatePimBool, 1 + needleIdxHost, pimIndividualNeedleMismatches[needleIdxPim]);
        assert (status == PIM_OK);
      }
      
      // Shift the haystack to the left to check the next character in it
      // Only shift if there will be another round of character checks in this iteration
      if(charIdx + 1 < needlesTableOrdering[iter].size()) {
        status = pimShiftElementsLeft(haystackPim);
        assert (status == PIM_OK);
      }
    }

    // Handle needles shorter than or equal to maxHammingDistance
    // These needles will match for all positions where it is fully in range of the haystack
    for(uint64_t needleIdx = 0; needleIdx < needlesTableOrdering[iter][0].size(); ++needleIdx) {
      uint64_t needleIdxHost = needlesTableOrdering[iter][0][needleIdx]; // Can be used to index into needles
      uint64_t needleIdxPim = (needlesTableActualToSorted[needleIdxHost] - needlesDone) + firstAvailPimNeedleResult; // Can be used to index into pimIndividualNeedleMismatches
      if(needles[needleIdxHost].size() <= maxHammingDistance) {
        // This string matches for all locations [0, haystack.size()-needle.size()]
        // Setup the match array for this string now, because we didn't create it earlier
        status = pimBroadcastUInt(pimIndividualNeedleMismatches[needleIdxPim], 1 + needleIdxHost);
        assert (status == PIM_OK);
        
        // Ensures that the last needle.size()-1 positions do not match
        for(uint64_t shiftIdx=1; shiftIdx < needles[needleIdxHost].size(); ++shiftIdx) {
          status = pimShiftElementsLeft(pimIndividualNeedleMismatches[needleIdxPim]);
          assert (status == PIM_OK);
        }
      }
    }

    // Update the final result array with the matches from this iteration
    // In the problem statement, we specify that the needle with the highest index should be given as a result if multiple match at the same position
    // Therefore, do a max reduction on the pimIndividualNeedleMismatches[needleIdx] objects
    for(uint64_t needleIdx = 1; needleIdx < needlesTableOrdering[iter][0].size() + firstAvailPimNeedleResult; ++needleIdx) {
      status = pimMax(pimIndividualNeedleMismatches[0], pimIndividualNeedleMismatches[needleIdx], pimIndividualNeedleMismatches[0]);
      assert (status == PIM_OK);
    }

    needlesDone += needlesTableOrdering[iter][0].size();

    if(iter + 1 < needlesTableOrdering.size()) {
      status = pimCopyObjectToObject(haystackCopyPim, haystackPim);
      assert (status == PIM_OK);
    }
  }

  status = pimCopyDeviceToHost(pimIndividualNeedleMismatches[0], (void *)matches.data());
  assert (status == PIM_OK);

  pimEndTimer();

  for(PimObjId individualNeedleMismatch : pimIndividualNeedleMismatches) {
    status = pimFree(individualNeedleMismatch);
    assert (status == PIM_OK);
  }

  status = pimFree(haystackPim);
  assert (status == PIM_OK);
  status = pimFree(intermediatePimBool);
  assert (status == PIM_OK);
  status = pimFree(isHaystackNonZeroPimBool);
  assert (status == PIM_OK);

  if(haystackCopyPim != -1) {
    status = pimFree(haystackCopyPim);
    assert (status == PIM_OK);
  }
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  
  std::cout << "Running PIM hamming string match" << std::endl;
  std::cout << "Keys file: " << params.keysInputFile << std::endl;
  std::cout << "Text file: " << params.textInputFile << std::endl;
  std::cout << "Hamming distance file: " << params.hammingDistanceInputFile << std::endl;
  
  std::string haystack;
  std::vector<std::string> needles;
  uint64_t maxHammingDistance = -1;
  std::vector<int> matches;

  haystack = readStringFromFile(params.textInputFile);
  needles = getNeedlesFromFile(params.keysInputFile);

  std::string hammingDistanceString = readStringFromFile(params.hammingDistanceInputFile);
  try {
    maxHammingDistance = std::stoull(hammingDistanceString);
  } catch (std::invalid_argument const& ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    exit(1);
  }
  
  
  if (!createDevice(params.configFile))
  {
    return 1;
  }

  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  assert(status == PIM_OK);

  matches.resize(haystack.size(), 0);

  NeedlesTable table = stringMatchPrecomputeTable(needles, 2 * deviceProp.numRowPerSubarray, deviceProp.isHLayoutDevice);

  hammingStringMatch(needles, haystack, maxHammingDistance, table, deviceProp.isHLayoutDevice, matches);

  if (params.shouldVerify) 
  {
    std::vector<int> matchesCpu;
    
    matchesCpu.resize(haystack.size());

    hammingStringMatchCpu(needles, haystack, maxHammingDistance, matchesCpu);

    // verify result
    bool ok = true;
    #pragma omp parallel for
    for (uint64_t i = 0; i < matches.size(); ++i)
    {
      if (matches[i] != matchesCpu[i])
      {
        #pragma omp critical
        {
          std::cerr << "Wrong answer: " << matches[i] << " (expected " << matchesCpu[i] << "), for position " << i << std::endl;
          ok = false;
        }
      }
    }
    if(ok) {
      std::cout << "Correct for hamming string match!" << std::endl;
    }
  }

  pimShowStats();

  return 0;
}


// References
// Cheng-Hung Lin, Chen-Hsiung Liu, Lung-Sheng Chien, Shih-Chieh Chang, "Accelerating Pattern Matching Using a Novel Parallel Algorithm on GPUs," IEEE Transactions on Computers, vol. 62, no. 10, pp. 1906-1916, Oct. 2013, doi:10.1109/TC.2012.254