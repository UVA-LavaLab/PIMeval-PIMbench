// Test: Data Generator for hamming string matching
// Copyright (c) 2025 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <unistd.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <iterator>
#include <random>
#include <iomanip>
#include <ios>
#include <vector>
#include <unordered_set>
#include <string>
#if defined(_OPENMP)
#include <omp.h>
#endif

typedef struct Params
{
  char *outputFile;
  size_t textLen;
  size_t numKeys;
  size_t minKeyLen;
  size_t maxKeyLen;
  size_t maxHammingDistance;
  uint8_t keyFrequency;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./data-generator.out [options]"
          "\n"
          "\n    -o    output folder name, stores output in dataset/[name]/text.txt, dataset/[name]/keys.txt, and dataset/[name]/maxHammingDistance.txt (required)"
          "\n    -l    length of text to match (default=10,000)"
          "\n    -n    number of keys (default=5)"
          "\n    -m    minimum key length (default = 1)"
          "\n    -x    maximum key length (default = 10)"
          "\n    -d    maximum hamming distance (default = 3)"
          "\n    -f    approximate frequency of keys in generated text, 0=no matches, 100=all keys (default = 50)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.outputFile = nullptr;
  p.textLen = 10000;
  p.numKeys = 5;
  p.minKeyLen = 1;
  p.maxKeyLen = 10;
  p.maxHammingDistance = 3;
  p.keyFrequency = 50;

  int opt;
  while ((opt = getopt(argc, argv, "h:o:l:n:m:x:d:f:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'o':
      p.outputFile = optarg;
      break;
    case 'l':
      p.textLen = strtoull(optarg, NULL, 0);
      break;
    case 'n':
      p.numKeys = strtoull(optarg, NULL, 0);
      break;
    case 'm':
      p.minKeyLen = strtoull(optarg, NULL, 0);
      break;
    case 'x':
      p.maxKeyLen = strtoull(optarg, NULL, 0);
      break;
    case 'd':
      p.maxHammingDistance = strtoull(optarg, NULL, 0);
      break;
    case 'f':
      p.keyFrequency = static_cast<uint8_t>(strtoull(optarg, NULL, 0));
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

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

int main(int argc, char* argv[])
{

  struct Params params = getInputParams(argc, argv);
  std::cout << "Generating Data..." << std::endl;

  if(params.outputFile == nullptr) {
    std::cerr << "Output filename must be provided!" << std::endl;
    return 1;
  }

  double maxPossibleKeys = 0;
  for (size_t length = params.minKeyLen; length <= params.maxKeyLen; ++length) {
    maxPossibleKeys += std::pow(26, length);
  }

  if((double) params.numKeys > maxPossibleKeys) {
    std::cerr << "Error: Number of keys greater than max possible keys for the given length range" << std::endl;
    std::cerr << "Requested Keys: " << params.numKeys << ", max number of unique keys of length [" << params.minKeyLen << ", " << params.maxKeyLen << "]: " << std::fixed << std::setprecision(0) << maxPossibleKeys << std::endl;
    return 1;
  }

  std::unordered_set<std::string> keysSet;
  keysSet.reserve(params.numKeys);

  // Create params.numKeys randomly generated keys of random sizes in range [params.minKeyLen, params.maxKeyLen]
  // All keys are unique
  while(keysSet.size() < params.numKeys) {
    size_t currKeyLen = rand()%(params.maxKeyLen-params.minKeyLen + 1) + params.minKeyLen;
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

  std::sort(keys.begin(), keys.end(), [](auto& l, auto& r){
    return l.size() < r.size();
  });

  // Stores indices of all keys to be put into the text
  std::vector<size_t> textVecOfKeys;
  
  size_t textCharsReplacedWithKeys = 0;
  size_t targetTextCharsReplacedWithKeys = ((double) params.keyFrequency / (double) 100.0) * params.textLen;
  targetTextCharsReplacedWithKeys = std::min(targetTextCharsReplacedWithKeys, params.textLen);
  size_t lastViableKey = keys.size();

  // Replace some of text with keys to generate matches
  while(textCharsReplacedWithKeys < targetTextCharsReplacedWithKeys && lastViableKey > 0) {
    size_t nextKeyInd = rand()%lastViableKey;
    std::string& nextKey = keys[nextKeyInd];
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
  && (keys[0].size() < params.textLen)) {
    textVecOfKeys.push_back(0);
    textCharsReplacedWithKeys = keys[0].size();
  }

  // Generate random string of text of length params.textLen, all of uppercase letters so that there aren't extra matches
  std::string text(params.textLen, 0);

  #pragma omp parallel
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist('A', 'Z');

    #pragma omp for
    for (size_t i = 0; i < params.textLen; ++i) {
      text[i] = dist(gen);
    }
  }

  std::random_device global_rd;
  std::mt19937 global_gen(global_rd());
  
  // Replace text with keys, approximately evenly spaced
  if(textVecOfKeys.size() == 1) {
    hammingTextReplace(text, 0, keys[0], params.maxHammingDistance, global_gen);
  } else if(textVecOfKeys.size() > 1) {
    size_t nonKeyCharsInText = params.textLen - textCharsReplacedWithKeys;
    size_t minSpace = nonKeyCharsInText / textVecOfKeys.size();
    size_t extraSpaces = nonKeyCharsInText % textVecOfKeys.size();

    size_t textInd = 0;
    for(size_t i=0; i < textVecOfKeys.size(); ++i) {
      std::string& currentKey = keys[textVecOfKeys[i]];
      hammingTextReplace(text, textInd, currentKey, params.maxHammingDistance, global_gen);
      textInd += currentKey.size();
      textInd += minSpace;
      if(extraSpaces > 0) {
        ++textInd;
        -- extraSpaces;
      }
    }
  }
  
  std::string outputFile(params.outputFile);
  std::string outputDir = "./../dataset/" + outputFile;
  std::string keyOutputFile = outputDir + "/keys.txt";
  std::string textOutputFile = outputDir + "/text.txt";
  std::string hammingDistanceOutputFile = outputDir + "/maxHammingDistance.txt";

  if (!std::filesystem::create_directory(outputDir)) {
    std::cerr << "Error creating output directory, dataset/" << params.outputFile << " may already exist" << std::endl;
    return 1;
  }

  FILE* textFile = fopen(textOutputFile.c_str(), "w");
  if (textFile == nullptr) {
    std::cerr << "Error opening text file" << std::endl;
    return 1;
  }

  size_t textWritten = fwrite(text.c_str(), sizeof(char), text.size(), textFile);
  if (textWritten != text.size()) {
    std::cerr << "Error writing to text file" << std::endl;
    return 1;
  }

  fclose(textFile);

  FILE* keysFile = fopen(keyOutputFile.c_str(), "w");
  if (keysFile == nullptr) {
    std::cerr << "Error opening keys file" << std::endl;
    return 1;
  }

  const std::string newlineChar = "\n";
  for(size_t i=0; i<keys.size(); ++i) {
    size_t keyWritten = fwrite(keys[i].c_str(), sizeof(char), keys[i].size(), keysFile);
    if (keyWritten != keys[i].size()) {
      std::cerr << "Error writing to keys file" << std::endl;
      return 1;
    }
    fwrite(newlineChar.c_str(), sizeof(char), newlineChar.size(), keysFile);
  }

  fclose(keysFile);

  std::string hammingDistanceString = std::to_string(params.maxHammingDistance);

  FILE* hammingDistanceFile = fopen(hammingDistanceOutputFile.c_str(), "w");
  if (hammingDistanceFile == nullptr) {
    std::cerr << "Error opening hamming distance file" << std::endl;
    return 1;
  }

  size_t hammingDistanceWritten = fwrite(hammingDistanceString.c_str(), sizeof(char), hammingDistanceString.size(), hammingDistanceFile);
  if (hammingDistanceWritten != hammingDistanceString.size()) {
    std::cerr << "Error writing to hamming distance file" << std::endl;
    return 1;
  }

  fclose(hammingDistanceFile);

  std::cout << "Successfully wrote to dataset/" << params.outputFile << "/keys.txt, dataset/"
            << params.outputFile << "/text.txt, and dataset/"
            << params.outputFile << "/maxHammingDistance.txt" << std::endl;
  
  return 0;
}
