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

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t stringLength;
  uint64_t keyLength;
  uint64_t numKeys;
  char *inputFile;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./string-match.out [options]"
          "\n"
          "\n    -s    string size (default=2048 elements)"
          "\n    -k    key size (default = 20 elements)"
          "\n    -n    number of keys (default = 4 keys)"
          "\n    -i    input file containing string and keys (default=generates strings with random characters)"
          "\n");
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.stringLength = 2048;
  p.keyLength = 20;
  p.numKeys = 4;
  p.inputFile = nullptr;

  int opt;
  while ((opt = getopt(argc, argv, "h:s:k:n:i:")) >= 0)
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
      p.inputFile = optarg;
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

// hs_database_t* compile_pattern(const char* haystack) {
//     hs_database_t *database;
//     hs_compile_error_t *compile_err;
//     if (hs_compile(haystack, HS_FLAG_DOTALL, HS_MODE_BLOCK, NULL, &database,
//                    &compile_err) != HS_SUCCESS) {
//         fprintf(stderr, "Hyperscan couldn't compile pattern, exiting\"%s\": %s\n",
//                 haystack, compile_err->message);
//         hs_free_compile_error(compile_err);
//         return nullptr;
//     }
// }

static int match_callback(unsigned int id, unsigned long long from,
                        unsigned long long to, unsigned int flags, void *matches) {
    printf("Match for pattern from: %llu, id: %u, to: %llu \n", from, id, to);
    // ((uint8_t*) ctx)[from] = 1;
    (*(vector<vector<uint8_t>>*)matches)[id][from] = 1;
    printf("set matches arr\n");
    return 0;
}

/**
 * @brief cpu string match kernel
 */
// int string_match_cpu(vector<string>& needles, string& haystack, vector<vector<uint8_t>>& matches, hs_database_t* db, hs_scratch_t* scratch) {

//   if(hs_scan(db, haystack.c_str(), haystack.size(), 0, scratch, match_callback, (void*) &matches) != HS_SUCCESS) {
//     fprintf(stderr, "Hyperscan couldn't scan the haystack, exiting\n");
//     hs_free_scratch(scratch);
//     hs_free_database(db);
//     return -1;
//   }

//   return 0;

//   // for(uint64_t needle_idx = 0; needle_idx < needles.size(); ++needle_idx) {
//   //   hs_database_t *database;
//   //   hs_compile_error_t *compile_err;

//   //   // TODO: Replace with hs_compile_lit_multi
//   //   if(hs_compile_lit(needles[needle_idx].c_str(), HS_FLAG_SOM_LEFTMOST, needles[needle_idx].size(), HS_MODE_BLOCK, NULL, &database, &compile_err) != HS_SUCCESS) {
//   //       fprintf(stderr, "Hyperscan couldn't compile pattern, exiting\"%s\": %s\n", haystack.c_str(), compile_err->message);
//   //       hs_free_compile_error(compile_err);
//   //       return;
//   //   }

//   //   hs_scratch_t *scratch = NULL;
//   //   if (hs_alloc_scratch(database, &scratch) != HS_SUCCESS) {
//   //       fprintf(stderr, "Hyperscan couldn't allocate scratch memory, exiting\n");
//   //       hs_free_database(database);
//   //       return;
//   //   }
//   //   // matches[needle_idx].data()
//   //   if(hs_scan(database, haystack.c_str(), haystack.size(), 0, scratch, match_callback, (void*) matches[needle_idx].data()) != HS_SUCCESS) {
//   //     fprintf(stderr, "Hyperscan couldn't scan the haystack, exiting\n");
//   //     hs_free_scratch(scratch);
//   //     hs_free_database(database);
//   //     return;
//   //   }
//   // }
// }

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
  if (params.inputFile == nullptr)
  {
    getString(haystack, params.stringLength);
    for(uint64_t i=0; i < params.numKeys; ++i) {
      needles.push_back("");
      getString(needles.back(), params.keyLength);
    }

    haystack = "dabcslkdfjdljed";
    needles = {"abc", "d", "z", "bcslk", "dabcs"};
  } 
  else 
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 1;
  }

  matches.resize(needles.size());

  for(uint64_t needle_idx = 0; needle_idx < needles.size(); ++needle_idx) {
    matches[needle_idx].resize(haystack.size());
  }

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
    ids_arr[needle_idx] = needle_idx;
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
  for(vector<uint8_t> match : matches) {
    printVec(match);
  }

  hs_free_database(db);
  hs_free_scratch(scratch);

  return 0;
}
