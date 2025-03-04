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
#include <unordered_set>
#include <filesystem>
#include <algorithm>                   // for max
#include <limits>                      // for numeric_limits

#if defined(_OPENMP)
#include <omp.h>
#endif


//! @brief  Reads a list of needles from an input file
//! @param[in]  keysInputFilename  The name of the file to read from
//! @return  A list of needles read from the file
//! @warning Exits program if file cannot be read from or is empty
std::vector<std::string> getNeedlesFromFile(const char* keysInputFilename) {
  std::ifstream keysFile(keysInputFilename);
  if (!keysFile) {
    std::cerr << "Error: Cannot open keys file " << keysInputFilename << ", aborting" << std::endl;
    std::exit(1);
  }
  
  std::vector<std::string> needles;
  std::string line;
  while (std::getline(keysFile, line)) {
    // As the output is formatted as the 32 bit index of the needle for matches, there cannot be more than INT_MAX needles
    if(needles.size() == std::numeric_limits<int>::max()) {
      std::cerr << "Error: Too many keys, can only process up to " << std::numeric_limits<int>::max() << " keys, aborting" << std::endl;
      std::exit(1);
    }

    needles.push_back(line);
  }
  
  if(needles.empty()) {
    std::cerr << "Error: Keys file (" << keysInputFilename << ") is empty, aborting" << std::endl;
    std::exit(1);
  }

  return needles;
}

//! @brief  Reads a string from a file
//! @param[in]  inputFileName  The name of the file to read from
//! @return  The string read from the file
//! @warning Exits program if file cannot be read from or is empty
std::string readStringFromFile(const char* inputFileName) {
  std::ifstream file(inputFileName);
  if (!file) {
    std::cerr << "Error: Cannot open file " << inputFileName << ", aborting" << std::endl;
    std::exit(1);
  }

  std::ostringstream fileOss;
  fileOss << file.rdbuf();
  std::string fileText = fileOss.str();
  if(fileText.empty()) {
    std::cerr << "Error: File is empty " << inputFileName << ", aborting" << std::endl;
    std::exit(1);
  }
  return fileText;
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
    #pragma omp parallel for
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

// Data Generator Functions ~~~~~~~~~~~~~~~~~~~~~~~~~

//! @brief  Replaces part of a text with random characters
//! @param[in,out]  text  The text to replace part of
//! @param[in]  idx  The start index to replace from
//! @param[in]  key  The key to insert into the text
//! @param[in]  maxHammingDistance  The maximum number of character changes to the key before insertion
//! @param[in]  gen  Random number generator
void hammingTextReplace(std::string& text, const uint64_t idx, const std::string& key, const uint64_t maxHammingDistance, std::mt19937& gen) {
  std::uniform_int_distribution<> numCharsDist(0, std::min(key.size(), maxHammingDistance));
  uint64_t numCharsToSwitch = numCharsDist(gen);
  std::uniform_int_distribution<> charsDist('a', 'b');
  for(uint64_t keyIdx=0; keyIdx < key.size(); ++keyIdx) {
    char nextChar;
    if(numCharsToSwitch) {
      nextChar = charsDist(gen);
      --numCharsToSwitch;
    } else {
      nextChar = key[keyIdx];
    }
    text[idx + keyIdx] = nextChar;
  }
}

//! @brief  Generates test data for string matching (Hamming or exact)
//! @param[in]  outputFolderName  output folder name, stores output in dataset/[name]/text.txt, dataset/[name]/keys.txt, and, if a hamming distance is provided, dataset/[name]/maxHammingDistance.txt
//! @param[in]  textLen  The length of text to match against
//! @param[in]  numKeys  The number of keys to generate
//! @param[in]  minKeyLen  Minimum key length to generate
//! @param[in]  maxKeyLen  Maximum key length to generate
//! @param[in]  keyFrequency  Approximate frequency of keys in generated text, 0=no matches, 100=all keys
//! @param[in]  isHamming  Whether or not to generate Hamming data
//! @param[in]  maxHammingDistance  The maximum Hamming distance that matches can have
void generateStringMatchData(const char* outputFolderName, const uint64_t textLen,
                             const uint64_t numKeys, const uint64_t minKeyLen,
                             const uint64_t maxKeyLen, const uint8_t keyFrequency,
                             const bool isHamming = false, const uint64_t maxHammingDistance = 0
                            ) {
  if(outputFolderName == nullptr) {
    std::cerr << "Output filename must be provided!" << std::endl;
    std::exit(1);
  }

  double maxPossibleKeys = 0;
  for (size_t length = minKeyLen; length <= maxKeyLen; ++length) {
    maxPossibleKeys += std::pow(26, length);
  }

  if((double) numKeys > maxPossibleKeys) {
    std::cerr << "Error: Number of keys greater than max possible keys for the given length range" << std::endl;
    std::cerr << "Requested Keys: " << numKeys << ", max number of unique keys of length [" << minKeyLen << ", " << maxKeyLen << "]: " << std::fixed << std::setprecision(0) << maxPossibleKeys << std::endl;
    std::exit(1);
  }

  std::unordered_set<std::string> keysSet;
  keysSet.reserve(numKeys);

  // Create numKeys randomly generated keys of random sizes in range [minKeyLen, maxKeyLen]
  // All keys are unique
  while(keysSet.size() < numKeys) {
    size_t currKeyLen = rand()%(maxKeyLen-minKeyLen + 1) + minKeyLen;
    std::string nextKey(currKeyLen, 0);
    for(size_t j=0; j<currKeyLen; ++j) {
      nextKey[j] = (rand() % 26) + 'a';
    }
    if(!keysSet.count(nextKey)) {
      keysSet.insert(nextKey);
    }
  }

  std::vector<std::string> keys(
        std::make_move_iterator(keysSet.begin()),
        std::make_move_iterator(keysSet.end())
    );

  std::vector<std::string> keysSorted(keys);

  std::sort(keysSorted.begin(), keysSorted.end(), [](auto& l, auto& r){
    return l.size() < r.size();
  });

  // Stores indices of all keys to be put into the text
  std::vector<size_t> textVecOfKeys;
  
  size_t textCharsReplacedWithKeys = 0;
  size_t targetTextCharsReplacedWithKeys = ((double) keyFrequency / (double) 100.0) * textLen;
  targetTextCharsReplacedWithKeys = std::min(targetTextCharsReplacedWithKeys, textLen);
  size_t lastViableKey = keysSorted.size();

  // Replace some of text with keys to generate matches
  while(textCharsReplacedWithKeys < targetTextCharsReplacedWithKeys && lastViableKey > 0) {
    size_t nextKeyInd = rand()%lastViableKey;
    std::string& nextKey = keysSorted[nextKeyInd];
    if(nextKey.size() + textCharsReplacedWithKeys > targetTextCharsReplacedWithKeys) {
      lastViableKey = nextKeyInd;
      continue;
    }
    textVecOfKeys.push_back(nextKeyInd);
    textCharsReplacedWithKeys += nextKey.size();
  }

  // Ensure that there is at least one key if the key frequency is non-zero
  // Covers edge case where all keys are too long to fit without going over desired frequency,
  // otherwise causing zero matches at low key frequencies
  if((textCharsReplacedWithKeys == 0 && targetTextCharsReplacedWithKeys > 0)
  && (keysSorted[0].size() < textLen)) {
    textVecOfKeys.push_back(0);
    textCharsReplacedWithKeys = keysSorted[0].size();
  }

  // Generate random string of text of length params.textLen, all of uppercase letters so that there aren't extra matches
  std::string text(textLen, 0);

  #pragma omp parallel
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist('A', 'Z');

    #pragma omp for
    for (size_t i = 0; i < textLen; ++i) {
      text[i] = dist(gen);
    }
  }

  std::random_device global_rd;
  std::mt19937 global_gen(global_rd());
  
  // Replace text with keys, approximately evenly spaced
  if(textVecOfKeys.size() == 1 && keysSorted[0].size() <= text.size()) {
    if(isHamming) {
      hammingTextReplace(text, 0, keysSorted[0], maxHammingDistance, global_gen);
    } else {
      text.replace(0, keys[0].size(), keys[0]);
    }
  } else if(textVecOfKeys.size() > 1) {
    size_t nonKeyCharsInText = textLen - textCharsReplacedWithKeys;
    size_t minSpace = nonKeyCharsInText / textVecOfKeys.size();
    size_t extraSpaces = nonKeyCharsInText % textVecOfKeys.size();

    size_t textInd = 0;
    for(size_t i=0; i < textVecOfKeys.size(); ++i) {
      std::string& currentKey = keysSorted[textVecOfKeys[i]];
      if(isHamming) {
        hammingTextReplace(text, textInd, currentKey, maxHammingDistance, global_gen);
      } else {
        text.replace(textInd, currentKey.size(), currentKey);
      }
      textInd += currentKey.size();
      textInd += minSpace;
      if(extraSpaces > 0) {
        ++textInd;
        -- extraSpaces;
      }
    }
  }
  
  std::string outputFileString(outputFolderName);
  std::string outputDir = "./../dataset/" + outputFileString;
  std::string keyOutputFile = outputDir + "/keys.txt";
  std::string textOutputFile = outputDir + "/text.txt";

  if (!std::filesystem::create_directory(outputDir)) {
    std::cerr << "Error creating output directory, dataset/" << outputFolderName << " may already exist" << std::endl;
    std::exit(1);
  }

  FILE* textFile = fopen(textOutputFile.c_str(), "w");
  if (textFile == nullptr) {
    std::cerr << "Error opening text file" << std::endl;
    std::exit(1);
  }

  size_t textWritten = fwrite(text.c_str(), sizeof(char), text.size(), textFile);
  if (textWritten != text.size()) {
    std::cerr << "Error writing to text file" << std::endl;
    std::exit(1);
  }

  fclose(textFile);

  FILE* keysFile = fopen(keyOutputFile.c_str(), "w");
  if (keysFile == nullptr) {
    std::cerr << "Error opening keys file" << std::endl;
    std::exit(1);
  }

  // Regular string matching requires keys to be in sorted order
  const std::vector<std::string>& keysToWrite = isHamming ? keys : keysSorted;

  const std::string newlineChar = "\n";
  for(size_t i=0; i<keysToWrite.size(); ++i) {
    size_t keyWritten = fwrite(keysToWrite[i].c_str(), sizeof(char), keysToWrite[i].size(), keysFile);
    if (keyWritten != keysToWrite[i].size()) {
      std::cerr << "Error writing to keys file" << std::endl;
      std::exit(1);
    }
    fwrite(newlineChar.c_str(), sizeof(char), newlineChar.size(), keysFile);
  }

  fclose(keysFile);

  if(isHamming) {
    std::string hammingDistanceOutputFile = outputDir + "/maxHammingDistance.txt";
    std::string hammingDistanceString = std::to_string(maxHammingDistance);

    FILE* hammingDistanceFile = fopen(hammingDistanceOutputFile.c_str(), "w");
    if (hammingDistanceFile == nullptr) {
      std::cerr << "Error opening hamming distance file" << std::endl;
      std::exit(1);
    }

    size_t hammingDistanceWritten = fwrite(hammingDistanceString.c_str(), sizeof(char), hammingDistanceString.size(), hammingDistanceFile);
    if (hammingDistanceWritten != hammingDistanceString.size()) {
      std::cerr << "Error writing to hamming distance file" << std::endl;
      std::exit(1);
    }

    fclose(hammingDistanceFile);

    std::cout << "Successfully wrote to dataset/" << outputFolderName << "/keys.txt, dataset/"
            << outputFolderName << "/text.txt, and dataset/"
            << outputFolderName << "/maxHammingDistance.txt" << std::endl;
  } else {
    std::cout << "Successfully wrote to dataset/" << outputFolderName << "/keys.txt and dataset/"
            << outputFolderName << "/text.txt" << std::endl;
  }
}

#endif