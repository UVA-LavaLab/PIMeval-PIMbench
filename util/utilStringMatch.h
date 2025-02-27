// Util: String matching utilities
// Copyright (c) 2025 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef PIM_FUNC_SIM_APPS_UTIL_STRING_MATCH_H
#define PIM_FUNC_SIM_APPS_UTIL_STRING_MATCH_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <stdint.h>
#include <algorithm>                   // for max
#include <limits>                      // for numeric_limits


//! @brief  Reads a list of needles from an input file
//! @param[in]  keysInputFilename  The name of the file to read from
//! @return  A list of needles read from the file
std::vector<std::string> getNeedlesFromFile(const char* keysInputFilename) {
  std::ifstream keysFile(keysInputFilename);
  if (!keysFile) {
    std::cerr << "Error: Cannot open keys file " << keysInputFilename << std::endl;
    std::exit(1);
  }
  
  std::vector<std::string> needles;
  std::string line;
  while (std::getline(keysFile, line)) {
    // As the output is formatted as the 32 bit index of the needle for matches, there cannot be more than INT_MAX needles
    if(needles.size() == std::numeric_limits<int>::max()) {
      std::cerr << "Error: Too many keys, can only process up to " << std::numeric_limits<int>::max() << " keys" << std::endl;
      std::exit(1);
    }

    needles.push_back(line);
  }
  
  if(needles.empty()) {
    std::cerr << "Error: Keys file (" << keysInputFilename << ") is empty " << std::endl;
    std::exit(1);
  }

  return needles;
}

//! @brief  Reads a string from a file
//! @param[in]  inputFileName  The name of the file to read from
//! @return  The string read from the file
std::string readStringFromFile(const char* inputFileName) {
  std::ifstream file(inputFileName);
  if (!file) {
    std::cerr << "Error: Cannot open file " << inputFileName << std::endl;
    std::exit(1);
  }

  std::ostringstream fileOss;
  fileOss << file.rdbuf();
  if(fileOss.eof()) {
    std::cerr << "Error: File is empty " << inputFileName << std::endl;
    std::exit(1);
  }
  return fileOss.str();
}

//! @brief  Matches a list of strings against another string on the CPU, with exact matching
//! @param[in]  needles  The list of needles to be matched
//! @param[in]  haystack  The string to match against
//! @param[out]  matches  A list of matches found
void stringMatchCpu(std::vector<std::string>& needles, std::string& haystack, std::vector<int>& matches) {
  for(uint64_t needleIdx = 0; needleIdx < needles.size(); ++needleIdx) {
    size_t pos = haystack.find(needles[needleIdx], 0);

    while (pos != std::string::npos) {
      matches[pos] = std::max(matches[pos], static_cast<int>(1 + needleIdx));
      pos = haystack.find(needles[needleIdx], pos + 1);
    }
  }
}

//! @brief  Matches a list of strings against another string on the CPU, with an allowable hamming distance
//! @param[in]  needles  The list of needles to be matched
//! @param[in]  haystack  The string to match against
//! @param[in]  maxHammingDistance  The maximum hamming distance that will allow a match
//! @param[out]  matches  A list of matches found
void hammingStringMatchCpu(const std::vector<std::string>& needles, const std::string& haystack, const uint64_t maxHammingDistance, std::vector<int>& matches) {
  for(uint64_t needleIdx = 0; needleIdx < needles.size(); ++needleIdx) {
    const std::string& needle = needles[needleIdx];
    if(needle.size() > haystack.size()) {
      continue;
    }
    for(uint64_t haystackIdx = 0; haystackIdx <= haystack.size()-needle.size(); ++haystackIdx) {
      uint64_t mismatches = 0;
      for(uint64_t charIdx = 0; charIdx < needle.size(); ++charIdx) {
        if(haystack[haystackIdx + charIdx] != needle[charIdx]) {
          ++mismatches;
          if(mismatches > maxHammingDistance) {
            break;
          }
        }
      }
      if(mismatches <= maxHammingDistance) {
        matches[haystackIdx] = std::max(matches[haystackIdx], static_cast<int>(1 + needleIdx));
      }
    }
  }
}

#endif