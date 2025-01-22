/**
 * @file string-match.cpp
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
#include <hs.h>

#include "utilBaselines.h"
#include "string-match-utils.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  char *keysInputFile;
  char *textInputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./string-match.out [options]"
          "\n"
          "\n    -k    keys input file, with each key on a seperate line (required, searches in cpp-string-match/dataset directory, note that keys are expected to be sorted by length, with smaller keys first)"
          "\n    -t    text input file to search for keys from (required, searches in cpp-string-match/dataset directory)"
          "\n    -v    t = verifies hyperscan output with host output. (default=false)"
          "\n");
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.keysInputFile = nullptr;
  p.textInputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:k:t:v:")) >= 0)
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

static int match_callback(unsigned int id, unsigned long long from,
                        unsigned long long to, unsigned int flags, void *matches) {
    // printf("Match for pattern from: %llu, id: %u, to: %llu \n", from, id, to);
    // ((uint8_t*) ctx)[from] = 1;
    (*(std::vector<int>*)matches)[from] = max(
      (*(std::vector<int>*)matches)[from]
      , (int) id);
    // printf("set matches arr\n");
    return 0;
}


template <typename T>
void printVec(std::vector<T>& vec) {
  for(T elem : vec) {
    cout << elem << ", ";
  }
  std::cout << std::endl;
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char **argv)
{
  struct Params params = input_params(argc, argv);

  if(params.keysInputFile == nullptr) {
    std::cout << "Please provide a keys input file" << std::endl;
    return 1;
  }
  if(params.textInputFile == nullptr) {
    std::cout << "Please provide a text input file" << std::endl;
    return 1;
  }

  std::cout << "Running CPU string match for \"" << params.keysInputFile << "\" as the keys file, and \"" << params.textInputFile << "\" as the text input file\n";
  std::string haystack;
  std::vector<std::string> needles;
  std::vector<int> matches;

  const std::string DATASET_FOLDER_PREFIX = "./../../../dataset/";

  haystack = get_text_from_file(DATASET_FOLDER_PREFIX, params.textInputFile);
  if(haystack.size() == 0) {
    std::cout << "There was an error opening the text file" << std::endl;
    return 1;
  }

  needles = get_needles_from_file(DATASET_FOLDER_PREFIX, params.keysInputFile);
  if(needles.size() == 0) {
    std::cout << "There was an error opening the keys file" << std::endl;
    return 1;
  }

  matches.resize(haystack.size());

  char** needles_arr = (char**) malloc(sizeof(char*) * needles.size());
  unsigned* flags_arr = (unsigned*) malloc(sizeof(unsigned) * needles.size());
  unsigned* ids_arr = (unsigned*) malloc(sizeof(unsigned) * needles.size());
  size_t* lens_arr = (size_t*) malloc(sizeof(size_t) * needles.size());
  unsigned num_elements = needles.size();
  unsigned mode = HS_MODE_BLOCK;
  hs_platform_info_t* platform = NULL;
  hs_database_t* db;
  hs_compile_error_t* compile_err;

  for(uint64_t needle_idx = 0; needle_idx < needles.size(); ++needle_idx) {
    needles_arr[needle_idx] = (char*) needles[needle_idx].c_str();
    flags_arr[needle_idx] = HS_FLAG_SOM_LEFTMOST;
    ids_arr[needle_idx] = 1 + needle_idx;
    lens_arr[needle_idx] = needles[needle_idx].size();
  }

  hs_error_t hs_err = hs_compile_lit_multi(needles_arr, flags_arr,
                      ids_arr, lens_arr, num_elements, mode, platform,
                      &db, &compile_err);

  if(hs_err != HS_SUCCESS) {
    fprintf(stderr, "Hyperscan couldn't compile pattern, ending program!\nError expression number: %d\nError text: \"%s\"", compile_err->expression, compile_err->message);
    hs_free_compile_error(compile_err);
    return -1;
  }

  // TODO: Check if always null
  hs_free_compile_error(compile_err);

  hs_scratch_t *scratch = NULL;
  if (hs_alloc_scratch(db, &scratch) != HS_SUCCESS) {
      fprintf(stderr, "Hyperscan couldn't allocate scratch memory, exiting\n");
      hs_free_database(db);
      return -1;
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (int32_t i = 0; i < WARMUP; i++)
  {
    if(hs_scan(db, haystack.c_str(), haystack.size(), 0, scratch, match_callback, (void*) &matches) != HS_SUCCESS) {
      fprintf(stderr, "Hyperscan couldn't scan the haystack, exiting\n");
      hs_free_scratch(scratch);
      hs_free_database(db);
      return -1;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsedTime = (end - start)/WARMUP;
  std::cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << std::endl;
  
  // printVec(matches);

  hs_free_database(db);
  hs_free_scratch(scratch);

  if (params.shouldVerify) 
  {
    std::vector<int> matches_cpu;
    
    matches_cpu.resize(haystack.size(), 0);

    string_match_cpu(needles, haystack, matches_cpu);

    // verify result
    bool is_correct = true;
    #pragma omp parallel for
    for (unsigned i = 0; i < matches.size(); ++i)
    {
      if (matches[i] != matches_cpu[i])
      {
        std::cout << "Wrong answer: " << unsigned(matches[i]) << " (expected " << unsigned(matches_cpu[i]) << "), for position " << i << std::endl;
        is_correct = false;
      }
    }
    if(is_correct) {
      std::cout << "Correct for string match!" << std::endl;
    }
  }

  return 0;
}
