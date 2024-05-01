#include <iostream>
#include <omp.h>
#include <vector>
using namespace std;


#ifndef _COMMON_H_
#define _COMMON_H_

#define WARMUP 4

#define MAX_NUMBER 1024

typedef 	uint64_t	u64;
typedef		int64_t		i64;
typedef		uint32_t	u32;
typedef		int32_t		i32;
typedef		uint16_t	u16;
typedef		int16_t		i16;
typedef		uint8_t		u8;
typedef		int8_t		i8;

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

/**
* @brief creates a vector with random values
* @param row number of rows in the matrix
* @param col number of columns in the matrix
*/
void initMatrix(u64 row,  u64 col, data_t** dataMat)
{
    // Providing a seed value
    srand((unsigned)time(NULL));
    for (u64 i = 0; i < row; i++)
    {
        for (u64 j = 0; j < col; j++) {
            dataMat[i][j] = rand() % (i+j+1);
        }
    }
}

#endif
