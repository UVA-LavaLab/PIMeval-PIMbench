#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <stdint.h>
#include <iomanip>
#include <chrono>
#include <time.h>

#include "../../../utilBaselines.h"

#define MY_RANGE 1000

using namespace std;


/**
 * @brief cpu database filtering kernel
 */


typedef struct Params
{
    uint64_t inVectorSize;
    uint64_t key;
    bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./filter.out [options]"
          "\n"
          "\n    -n    database size (default=65536 elements)"
          "\n    -k    value of key (default = 1)"
          "\n    -v    t = print output vector. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.inVectorSize = 65536;
  p.key = 10;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:n:k:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'n':
      p.inVectorSize = strtoull(optarg, NULL, 0);
      break;
    case 'k':
      p.key = strtoull(optarg, NULL, 0);
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

/**
 * @brief Main of the Host Application.
 */

// static const uint64_t inVectorSize = 1073741824;

int main(int argc, char **argv){

    struct Params p = getInputParams(argc, argv);

    uint64_t inVectorSize = p.inVectorSize;

    vector<uint64_t> inVector(inVectorSize);

    std::cout << "DB element size: " << inVectorSize << std::endl;

    srand((unsigned)time(NULL));
    for (uint64_t i = 0; i < inVectorSize; i++){
        inVector[i] = rand() % MY_RANGE;
    }

    uint64_t dummyVectorSize = 1073741824;
    vector<int> dummyVector1(dummyVectorSize, 0);
    uint64_t buffer_in_CPU = 0;
    uint64_t selectedNum = 0;

    // Flushing the cache
    for (uint64_t j = 0; j < dummyVectorSize; j++){
      dummyVector1[j] += rand() % MY_RANGE;
    }

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedTime;
    for (uint64_t i = 0; i < inVectorSize; i++){
      if(p.key > inVector[i]){
        buffer_in_CPU += inVector[i];
        selectedNum++;
      }
    }
    uint64_t outSize = selectedNum;
    auto end = std::chrono::high_resolution_clock::now();
    elapsedTime = (end - start);

    if(p.shouldVerify == true){
      cout << outSize <<" out of " << inVectorSize << " selected" << endl;
    }
    cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;
    cout << endl;

    return 0;
}
