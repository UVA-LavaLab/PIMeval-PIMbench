/**
 * @file string-match.cu
 * @brief Template for a Host Application Source File.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <stdint.h>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <vector>
#include <memory>

#include "utilBaselines.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t stringLength;
  uint64_t keyLength;
  uint64_t numKeys;
  char *keysInputFile;
  char *textInputFile;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./string-match.out [options]"
          "\n"
          "\n    -s    string size (default=2048 elements)"
          "\n    -k    key size (default = 20 elements)"
          "\n    -n    number of keys (default = 4 keys)"
          "\n    -i    input file containing keys to search for (default=generates keys with random characters)"
          "\n    -t    input file containing string to search in (default=generates random string)"
          "\n");
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.stringLength = 2048;
  p.keyLength = 20;
  p.numKeys = 4;
  p.keysInputFile = nullptr;
  p.textInputFile = nullptr;

  int opt;
  while ((opt = getopt(argc, argv, "h:s:k:n:i:t:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 's':
      p.stringLength = strtoull(optarg, NULL, 0);
      break;
    case 'k':
      p.keyLength = strtoull(optarg, NULL, 0);
      break;
    case 'n':
      p.numKeys = strtoull(optarg, NULL, 0);
      break;
    case 'i':
      p.keysInputFile = optarg;
      break;
    case 't':
      p.textInputFile = optarg;
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

void getString(string& str, uint64_t len) {
  str.resize(len);
#pragma omp parallel for
  for(uint64_t i=0; i<len; ++i) {
    str[i] = 'a' + (rand()%26);
  }
}

void printVec(vector<uint8_t>& vec) {
  for(auto match : vec) {
    cout << (unsigned) (match) << ", ";
  }
  cout << endl;
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char **argv)
{
  struct Params params = input_params(argc, argv);

  std::cout << "Running CPU string match for string size: " << params.stringLength << ", key size: " << params.keyLength << ", number of keys: " << params.numKeys << "\n";
  string haystack;
  vector<string> needles;
  vector<vector<uint8_t>> matches;

  if (params.keysInputFile == nullptr)
  {
    for(uint64_t i=0; i < params.numKeys; ++i) {
      needles.push_back("");
      getString(needles.back(), params.keyLength);
    }
  } 
  else 
  {
    int keys_fd = open(params.keysInputFile, O_RDONLY);
    if (keys_fd < 0) 
    {
      perror("Failed to open keys input file, or file doesn't exist");
      return 1;
    }

    struct stat keys_finfo;
    if (fstat(keys_fd, &keys_finfo) < 0) 
    {
      perror("Failed to get file info for keys");
      return 1;
    }

    // char* keys_fdata = (char*) malloc(sizeof(char) * (keys_finfo.st_size + 1));
    string keys_fdata(keys_finfo.st_size, '\0');
    // if(keys_fdata == NULL) {
    //     perror("Failed to allocate memory for keys");
    //     return 1;
    // }

    // keys_fdata[keys_finfo.st_size] = '\0';
    if (read(keys_fd, keys_fdata.data(), keys_finfo.st_size) < 0) 
    {
      perror("Failed to read the keys file");
      return 1;
    }
    std::istringstream keys_fdata_stream(keys_fdata);
    std::string key_line;

    while (getline(keys_fdata_stream, key_line)) {
        needles.push_back(key_line);
    }

    cout << keys_fdata << endl;
  }

  if (params.textInputFile == nullptr)
  {
    getString(haystack, params.stringLength);
  } 
  else 
  {
    int haystack_fd = open(params.textInputFile, O_RDONLY);
    if (haystack_fd < 0) 
    {
      perror("Failed to open text input file, or file doesn't exist");
      return 1;
    }

    struct stat haystack_finfo;
    if (fstat(haystack_fd, &haystack_finfo) < 0) 
    {
      perror("Failed to get file info for text");
      return 1;
    }

    string haystack_fdata(haystack_finfo.st_size, '\0');

    haystack_fdata[haystack_finfo.st_size] = '\0';
    if (read(haystack_fd, haystack_fdata.data(), haystack_finfo.st_size) < 0) 
    {
      perror("Failed to read the text file");
      return 1;
    }
    cout << haystack_fdata << endl;
  }

  matches.resize(needles.size());

  for(uint64_t needle_idx = 0; needle_idx < needles.size(); ++needle_idx) {
    matches[needle_idx].resize(haystack.size());
  }

  return 0;
}
