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
          "\n    -k    keys input file, each key on new line (default=dataset/10mil_l-10_nk-10_kl/keys.txt) must be sorted by increasing length, must have a blank line at end of file"
          "\n    -t    text input file to search for keys from (default=dataset/10mil_l-10_nk-10_kl/text.txt)"
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
// Instead of calling pimEQScalar multiple times for the same character, call only once per character and reuse result
// Not included in benchmarking time, as it only depends on the keys/needles and not on the text/haystack
std::vector<std::vector<std::vector<size_t>>> stringMatchPrecomputeTable(std::vector<std::string>& needles, uint64_t numRows, bool isHorizontal) {
  std::vector<std::vector<std::vector<size_t>>> resultTable;
  
  // If vertical, each pim object takes 32 rows, 1 row if horizontal
  // Two rows used by the haystack and intermediate
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
  
  // Stores the text that is being checked for the needles
  PimObjId haystackPim = pimAlloc(PIM_ALLOC_AUTO, haystack.size(), PIM_UINT32);
  assert(haystackPim != -1);
  
  // Used for intermediate calculations
  PimObjId intermediatePim = pimAllocAssociated(haystackPim, PIM_UINT32);
  assert(intermediatePim != -1);

  // PIM simulator currently only supports operations between objects of the same size
  // We define the output as an array of 32 bit ints to represent the index of the keys
  // Therefore, all calculations must be done in 32 bits, so we cast everything to 32 bits
  // This can be removed when the simulator supports mixed width operands
  // TODO: remove when pim conversion becomes possible
  uint32_t *haystack32Bit = (uint32_t*) malloc(sizeof(uint32_t) * haystack.size());
  #pragma omp parallel for
  for(uint32_t i=0; i<haystack.size(); ++i) {
    haystack32Bit[i] = (uint32_t) haystack[i];
  }

  // Matches are calculated for a group of needles at a time, this vector stores the matches for each needle
  // needlesTable[0][0].size() is the number of needles in the first iteration, which will have the most needles out of all iterations

  if(needlesTable.size() == 0 || needlesTable[0].size() == 0) {
    std::cerr << "Error: The needles table is empty" << std::endl;
    exit(1);
  }

  std::vector<PimObjId> pimIndividualNeedleMatches;
  for(uint32_t i=0; i<needlesTable[0][0].size(); ++i) {
    pimIndividualNeedleMatches.push_back(pimAllocAssociated(haystackPim, PIM_UINT32));
    assert(pimIndividualNeedleMatches.back() != -1);
  }

  uint64_t needlesDone = 0;

  pimStartTimer();

  PimStatus status;
  // Number of needles at a time is limited by how many PIM objects can be alloc associated with each other in a subarray
  // Iterates multiple times if there are enough needles
  for(uint64_t iter=0; iter<needlesTable.size(); ++iter) {
    // Haystack is shifted throughout each iteration, this resets it in between iterations
    // Could be replaced with a copy object to object once available
    status = pimCopyHostToDevice((void *)haystack32Bit, haystackPim);
    assert (status == PIM_OK);

    // Instead of creating a seperate result variable, we reuse one of the objects in pimIndividualNeedleMatches
    // Can be used as a needle match array for only the first iteration, and a result array for the rest
    // Slight optimization allows one extra needle in the first iteration
    uint64_t firstAvailPimNeedleResult = iter == 0 ? 0 : 1;

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
        uint64_t needleIdxPim = (needleIdxHost - needlesDone) + firstAvailPimNeedleResult; // Can be used to index into pimIndividualNeedleMatches
        char currentChar = needles[needleIdxHost][charIdx];

        if(charIdx == 0) {
          // If on the first character index, there is no need to pimAnd with the current possible matches
          // Instead, place the equality result directly into the match array
          status = pimEQScalar(haystackPim, pimIndividualNeedleMatches[needleIdxPim], (uint64_t) currentChar);
          assert (status == PIM_OK);
        } else if(prevChar == currentChar) {
          // Reuse the previously calculated equality result in intermediatePim and pimAnd with the current matches
          status = pimAnd(pimIndividualNeedleMatches[needleIdxPim], intermediatePim, pimIndividualNeedleMatches[needleIdxPim]);
          assert (status == PIM_OK);
        } else {
          // Check the entirety of the text if it is equal with the current character
          status = pimEQScalar(haystackPim, intermediatePim, (uint64_t) currentChar);
          assert (status == PIM_OK);

          // Update the potential match array
          status = pimAnd(pimIndividualNeedleMatches[needleIdxPim], intermediatePim, pimIndividualNeedleMatches[needleIdxPim]);
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
      uint64_t needleIdxPim = needleIdx + firstAvailPimNeedleResult; // Can be used to index into pimIndividualNeedleMatches

      // pimIndividualNeedleMatches[needleIdxPim] is a binary PIM object containing only 0s and 1s
      // 0 in pimIndividualNeedleMatches[needleIdxPim] represents no match at the location, while a 1 is a match
      // To get the output, we want a 0 if there is a no match, or the 1-based index of the needle if there is a match
      // First, we replace all 1s in each pimIndividualNeedleMatches[needleIdxPim] with 1 + needleIdxHost, and leave all 0s
      // This can be done in two ways:
      //      - Multiply by 1 + needleIdxHost (1 * (1 + needleIdxHost) = 1 + needleIdxHost, 0 * (1 + needleIdxHost) = 0)
      //      - Binary Manipulation:
      //          - Invert 0s and 1s (pimXorScalar)
      //          - Subtract 1 (1->0, 0->-1==all ones - uses unsigned integer overflow)
      //          - And with 1 + needleIdxHost
      //
      // Both of these methods have the same output, however the multiplication will be faster on bit parallel architectures
      // The binary manipulation method is faster on bit serial architectures
      // On the bit parallel architectures tested, each of the operations above took the same time (multiplication, xor, etc.)
      // Therefore, on the bit parallel architectures, it is faster to do the single operation (multiplication)
      // However, on the bit serial architecture tested, multiplication is significantly slower because it is O(n^2) where n is the number of bits
      // Thus the second method is faster on the bit serial architecture tested
      // The isHorizonatal parameter serves as a simple check for which we are on
      if(isHorizontal) {
        status = pimMulScalar(pimIndividualNeedleMatches[needleIdxPim], pimIndividualNeedleMatches[needleIdxPim], 1 + needleIdxHost);
        assert (status == PIM_OK);
      } else {
        status = pimXorScalar(pimIndividualNeedleMatches[needleIdxPim], pimIndividualNeedleMatches[needleIdxPim], 1);
        assert (status == PIM_OK);

        status = pimSubScalar(pimIndividualNeedleMatches[needleIdxPim], pimIndividualNeedleMatches[needleIdxPim], 1);
        assert (status == PIM_OK);

        status = pimAndScalar(pimIndividualNeedleMatches[needleIdxPim], pimIndividualNeedleMatches[needleIdxPim], 1 + needleIdxHost);
        assert (status == PIM_OK);
      }
    }

    // Update the final result array with the matches from this iteration
    // In the problem statement, we specify that the longest needle should be given as a result if multiple match at the same position
    // needleIdxHost will be larger if the needle is longer, because we specify that needles should be sorted ahead of time
    // Therefore, to get the longest needle at each position, we do a max reduction on the pimIndividualNeedleMatches[needleIdx] objects
    for(uint64_t needleIdx = 1; needleIdx < needlesTable[iter][0].size() + firstAvailPimNeedleResult; ++needleIdx) {
      status = pimMax(pimIndividualNeedleMatches[0], pimIndividualNeedleMatches[needleIdx], pimIndividualNeedleMatches[0]);
      assert (status == PIM_OK);
    }

    needlesDone += needlesTable[iter][0].size();
  }

  status = pimCopyDeviceToHost(pimIndividualNeedleMatches[0], (void *)matches.data());
  assert (status == PIM_OK);

  pimEndTimer();

  free(haystack32Bit);

  status = pimFree(haystackPim);
  assert (status == PIM_OK);

  status = pimFree(intermediatePim);
  assert (status == PIM_OK);

  for(PimObjId individualNeedleMatch : pimIndividualNeedleMatches) {
    status = pimFree(individualNeedleMatch);
    assert (status == PIM_OK);
  }
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);

  const std::string defaultTextFileName = "./../dataset/10mil_l-10_nk-10_kl/text.txt";

  std::string textFilename;
  if(params.textInputFile == nullptr) {
    textFilename = defaultTextFileName;
  } else {
    textFilename = params.textInputFile;
  }

  const std::string defaultNeedlesFileName = "./../dataset/10mil_l-10_nk-10_kl/keys.txt";

  std::string needlesFilename;
  if(params.keysInputFile == nullptr) {
    needlesFilename = defaultNeedlesFileName;
  } else {
    needlesFilename = params.keysInputFile;
  }
  
  std::cout << "Running PIM string match for \"" << needlesFilename << "\" as the keys file, and \"" << textFilename << "\" as the text input file\n";
  
  std::string haystack;
  std::vector<std::string> needles;
  std::vector<int> matches;

  haystack = getTextFromFile(textFilename);
  if(haystack.size() == 0) {
    std::cerr << "There was an error opening the text file" << std::endl;
    return 1;
  }

  needles = getNeedlesFromFile(needlesFilename);
  if(needles.size() == 0) {
    std::cerr << "There was an error opening the keys file" << std::endl;
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
