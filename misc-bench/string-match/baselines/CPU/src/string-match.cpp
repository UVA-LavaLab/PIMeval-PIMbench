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
#include "utilStringMatch.h"

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
          "\n    -k    keys input file, each key on new line (default=dataset/10mil_l-10_nk-10_kl/keys.txt) must be sorted by increasing length, must have a blank line at end of file"
          "\n    -t    text input file to search for keys from (default=dataset/10mil_l-10_nk-10_kl/text.txt)"
          "\n    -v    t = verifies hyperscan output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
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

static int matchCallback(unsigned int id, unsigned long long from,
                        unsigned long long to, unsigned int flags, void *matches) {
    (*(std::vector<int>*)matches)[from] = max(
      (*(std::vector<int>*)matches)[from]
      , (int) id);
    return 0;
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char **argv)
{
  struct Params params = getInputParams(argc, argv);

  const std::string defaultTextFileName = "./../../../dataset/10mil_l-10_nk-10_kl/text.txt";

  std::string textFilename;
  if(params.textInputFile == nullptr) {
    textFilename = defaultTextFileName;
  } else {
    textFilename = params.textInputFile;
  }

  const std::string defaultNeedlesFileName = "./../../../dataset/10mil_l-10_nk-10_kl/keys.txt";

  std::string needlesFilename;
  if(params.keysInputFile == nullptr) {
    needlesFilename = defaultNeedlesFileName;
  } else {
    needlesFilename = params.keysInputFile;
  }

  std::cout << "Running CPU string match for \"" << needlesFilename << "\" as the keys file, and \"" << textFilename << "\" as the text input file\n";
  std::string haystack;
  std::vector<std::string> needles;
  std::vector<int> matches;

  haystack = readStringFromFile(textFilename.c_str());
  needles = getNeedlesFromFile(needlesFilename.c_str());

  matches.resize(haystack.size());

  char** needlesArr = (char**) malloc(sizeof(char*) * needles.size());
  unsigned* flagsArr = (unsigned*) malloc(sizeof(unsigned) * needles.size());
  unsigned* idsArr = (unsigned*) malloc(sizeof(unsigned) * needles.size());
  size_t* lensArr = (size_t*) malloc(sizeof(size_t) * needles.size());
  unsigned numElements = needles.size();
  unsigned mode = HS_MODE_BLOCK;
  hs_platform_info_t* platform = NULL;
  hs_database_t* db;
  hs_compile_error_t* compileErr;

  for(uint64_t needleIdx = 0; needleIdx < needles.size(); ++needleIdx) {
    needlesArr[needleIdx] = (char*) needles[needleIdx].c_str();
    flagsArr[needleIdx] = HS_FLAG_SOM_LEFTMOST;
    idsArr[needleIdx] = 1 + needleIdx;
    lensArr[needleIdx] = needles[needleIdx].size();
  }

  hs_error_t hsErr = hs_compile_lit_multi(needlesArr, flagsArr,
                      idsArr, lensArr, numElements, mode, platform,
                      &db, &compileErr);

  if(hsErr != HS_SUCCESS) {
    fprintf(stderr, "Hyperscan couldn't compile pattern, ending program!\nError expression number: %d\nError text: \"%s\"", compileErr->expression, compileErr->message);
    hs_free_compile_error(compileErr);
    return -1;
  }

  hs_free_compile_error(compileErr);

  hs_scratch_t *scratch = NULL;
  if (hs_alloc_scratch(db, &scratch) != HS_SUCCESS) {
      fprintf(stderr, "Hyperscan couldn't allocate scratch memory, exiting\n");
      hs_free_database(db);
      return -1;
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (int32_t i = 0; i < WARMUP; i++)
  {
    if(hs_scan(db, haystack.c_str(), haystack.size(), 0, scratch, matchCallback, (void*) &matches) != HS_SUCCESS) {
      fprintf(stderr, "Hyperscan couldn't scan the haystack, exiting\n");
      hs_free_scratch(scratch);
      hs_free_database(db);
      return -1;
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsedTime = (end - start)/WARMUP;
  std::cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << std::endl;

  hs_free_database(db);
  hs_free_scratch(scratch);

  free(needlesArr);
  free(flagsArr);
  free(idsArr);
  free(lensArr);

  if (params.shouldVerify) 
  {
    std::vector<int> matchesCpu;
    
    matchesCpu.resize(haystack.size(), 0);

    stringMatchCpu(needles, haystack, matchesCpu);

    // verify result
    bool ok = true;
    #pragma omp parallel for
    for (unsigned i = 0; i < matches.size(); ++i)
    {
      if (matches[i] != matchesCpu[i])
      {
        std::cout << "Wrong answer: " << unsigned(matches[i]) << " (expected " << unsigned(matchesCpu[i]) << "), for position " << i << std::endl;
        ok = false;
      }
    }
    if(ok) {
      std::cout << "Correct for string match!" << std::endl;
    }
  }

  return 0;
}
