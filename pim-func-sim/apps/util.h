#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;


#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef DATA_TYPE
typedef int32_t data_t;
#else
typedef DATA_TYPE data_t;
#endif


void getVector(uint64_t vectorLength, std::vector<int>& srcVector) {
  srand((unsigned)time(NULL));
  srcVector.reserve(vectorLength);
  #pragma omp parallel for
  for (int i = 0; i < vectorLength; ++i)
  {
    srcVector[i] = rand() % (i+1);
  }
}

#endif
