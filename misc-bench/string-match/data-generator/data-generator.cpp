// Data Generator for string matching
// Copyright (c) 2025 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <unistd.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <unordered_set>
#include <iterator>
#include <random>

typedef struct Params
{
  char *outputFile;
  size_t textLen;
  size_t numKeys;
  size_t minKeyLen;
  size_t maxKeyLen;
  uint8_t keyFrequency;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./data-generator.out [options]"
          "\n"
          "\n    -o    output folder name, stores output in dataset/[name]/text.txt and dataset/[name]/keys.txt (must be provided)"
          "\n    -l    length of text to match (default=10,000)"
          "\n    -n    number of keys (default=5)"
          "\n    -m    minimum key length (default = 1)"
          "\n    -x    maximum key length (default = 10)"
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
  p.keyFrequency = 50;

  int opt;
  while ((opt = getopt(argc, argv, "h:o:l:n:m:x:f:")) >= 0)
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
    case 'f':
      p.keyFrequency = (uint8_t) strtoull(optarg, NULL, 0);
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

int main(int argc, char* argv[])
{

  struct Params params = getInputParams(argc, argv);
  std::cout << "Generating Data..." << std::endl;

  if(params.outputFile == nullptr) {
    printf("Output filename must be provided!\n");
    return 1;
  }

  double maxPossibleKeys = 0;
  for (size_t length = params.minKeyLen; length <= params.maxKeyLen; ++length) {
    maxPossibleKeys += std::pow(26, length);
  }

  if((double) params.numKeys > maxPossibleKeys) {
    printf("Error: Number of keys greater than max possible keys for the given length range\n");
    printf("Requested Keys: %lu, max number of unique keys of length [%lu, %lu]: %.0f\n", params.numKeys, params.minKeyLen, params.maxKeyLen, maxPossibleKeys);
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
  size_t lastViableKey = keys.size();

  // Replace some of text with keys to generate matches
  while(textCharsReplacedWithKeys < targetTextCharsReplacedWithKeys && lastViableKey > 0) {
    size_t nextKeyInd = rand()%lastViableKey;
    std::string& nextKey = keys[nextKeyInd];
    if(nextKey.size() + textCharsReplacedWithKeys > targetTextCharsReplacedWithKeys) {
      lastViableKey = nextKeyInd;
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

  // Replace text with keys, approximately evenly spaced
  if(textVecOfKeys.size() == 1) {
    text.replace(0, keys[0].size(), keys[0]);
  } else if(textVecOfKeys.size() > 1) {
    size_t nonKeyCharsInText = params.textLen - textCharsReplacedWithKeys;
    size_t minSpace = nonKeyCharsInText / (textVecOfKeys.size() - 1);
    size_t extraSpaces = nonKeyCharsInText % (textVecOfKeys.size() - 1);

    size_t textInd = 0;
    for(size_t i=0; i < textVecOfKeys.size() - 1; ++i) {
      std::string& currentKey = keys[textVecOfKeys[i]];
      text.replace(textInd, currentKey.size(), currentKey);
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

  if (!std::filesystem::create_directory(outputDir)) {
      printf("Error creating output directory, dataset/%s may already exist\n", params.outputFile);
      return 1;
  }

  FILE* textFile = fopen(textOutputFile.c_str(), "w");
  if (textFile == nullptr) {
      printf("Error opening text file\n");
      return 1;
  }

  size_t textWritten = fwrite(text.c_str(), sizeof(char), text.size(), textFile);
  if (textWritten != text.size()) {
      printf("Error writing to text file\n");
      return 1;
  }

  fclose(textFile);

  FILE* keysFile = fopen(keyOutputFile.c_str(), "w");
  if (keysFile == nullptr) {
      printf("Error opening keys file\n");
      return 1;
  }

  std::string newlineChar = "\n";
  for(size_t i=0; i<keys.size(); ++i) {
    size_t keyWritten = fwrite(keys[i].c_str(), sizeof(char), keys[i].size(), keysFile);
    if (keyWritten != keys[i].size()) {
        printf("Error writing to keys file\n");
        return 1;
    }
    fwrite(newlineChar.c_str(), sizeof(char), newlineChar.size(), keysFile);
  }

  fclose(keysFile);

  printf("Successfully wrote to dataset/%s/keys.txt and dataset/%s/text.txt\n", params.outputFile, params.outputFile);

  return 0;
}
