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
          "\n    -f    approximate frequency of keys in generated text, 0=fully random text, 100=all keys (default = 50)"
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

  std::vector<std::string> keys;
  keys.reserve(params.numKeys);

  // Create params.numKeys randomly generated keys of random sizes in range [params.minKeyLen, params.maxKeyLen]
  for(size_t i=0; i<params.numKeys; ++i) {
    size_t curr_key_len = rand()%(params.maxKeyLen-params.minKeyLen + 1) + params.minKeyLen;
    keys.emplace_back(curr_key_len, 0);
    for(size_t j=0; j<curr_key_len; ++j) {
        keys.back()[j] = (rand() % 26) + 'a';
    }
  }

  std::sort(keys.begin(), keys.end(), [](auto& l, auto& r){
    return l.size() < r.size();
  });

  // Stores indices of all keys to be put into the text
  std::vector<size_t> text_vec_of_keys;
  
  size_t text_chars_replaced_with_keys = 0;
  size_t target_text_chars_replaced_with_keys = ((double) params.keyFrequency / (double) 100.0) * params.textLen;
  size_t last_viable_key = keys.size();

  // Replace some of text with keys to generate matches
  while(text_chars_replaced_with_keys < target_text_chars_replaced_with_keys && last_viable_key > 0) {
    size_t next_key_ind = rand()%last_viable_key;
    std::string& next_key = keys[next_key_ind];
    if(next_key.size() + text_chars_replaced_with_keys > target_text_chars_replaced_with_keys) {
      last_viable_key = next_key_ind;
    }
    // text.replace(text_chars_replaced_with_keys, next_key.size(), next_key);
    text_vec_of_keys.push_back(next_key_ind);
    text_chars_replaced_with_keys += next_key.size();
  }

  // Ensure that there is at least one key if the key frequency is non-zero
  // Covers edge case where all keys are too long to fit without going over desired frequency,
  // otherwise causing zero matches at low key frequencies
  if((text_chars_replaced_with_keys == 0 && target_text_chars_replaced_with_keys > 0)
  && (keys[0].size() < params.textLen)) {
    // text.replace(0, keys[0].size(), keys[0]);
    text_vec_of_keys.push_back(0);
    text_chars_replaced_with_keys = keys[0].size();
  }

  // Generate random string of text of length params.textLen
  std::string text(params.textLen, 0);

  for(size_t i=0; i<params.textLen; ++i) {
    text[i] = (rand() % 26) + 'A';
  }

  if(text_vec_of_keys.size() == 1) {
    text.replace(0, keys[0].size(), keys[0]);
  } else if(text_vec_of_keys.size() > 1) {
    size_t non_key_chars_in_text = params.textLen - text_chars_replaced_with_keys;
    size_t min_space = non_key_chars_in_text / (text_vec_of_keys.size() - 1);
    size_t extra_spaces = non_key_chars_in_text % (text_vec_of_keys.size() - 1);

    size_t text_ind = 0;
    for(size_t i=0; i < text_vec_of_keys.size() - 1; ++i) {
      std::string& current_key = keys[text_vec_of_keys[i]];
      text.replace(text_ind, current_key.size(), current_key);
      text_ind += current_key.size();
      text_ind += min_space;
      if(extra_spaces > 0) {
        ++text_ind;
        -- extra_spaces;
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

  FILE* text_file = fopen(textOutputFile.c_str(), "w");
  if (text_file == nullptr) {
      printf("Error opening text file\n");
      return 1;
  }

  size_t text_written = fwrite(text.c_str(), sizeof(char), text.size(), text_file);
  if (text_written != text.size()) {
      printf("Error writing to text file\n");
      return 1;
  }

  fclose(text_file);

  FILE* keys_file = fopen(keyOutputFile.c_str(), "w");
  if (text_file == nullptr) {
      printf("Error opening keys file\n");
      return 1;
  }

  std::string newline_char = "\n";
  for(size_t i=0; i<keys.size(); ++i) {
    size_t key_written = fwrite(keys[i].c_str(), sizeof(char), keys[i].size(), keys_file);
    if (key_written != keys[i].size()) {
        printf("Error writing to keys file\n");
        return 1;
    }
    if(i+1 < keys.size()) {
      fwrite(newline_char.c_str(), sizeof(char), newline_char.size(), keys_file);
    }
  }

  fclose(keys_file);

  printf("Successfully wrote to dataset/%s/keys.txt and dataset/%s/text.txt\n", params.outputFile, params.outputFile);

  return 0;
}
