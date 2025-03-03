// Test: C++ version of vector multiplication
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <getopt.h>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <random>
#include <set>
#include <smmintrin.h>
#include <stdint.h>
#include <string>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../../spmv-util.cpp"
#include "libpimeval.h"

using namespace std;

#define BLOCK_LEN 8

// Params ---------------------------------------------------------------------
typedef struct Params {
  uint64_t nnz;
  uint64_t matRowsBlocks, matColsBlocks;
  uint64_t maxBlocksPerRow, minBlocksPerRow;
  uint64_t maxNNZPerBlock, minNNZPerBlock;
  uint64_t maxVal;
  uint64_t seed;
  uint64_t batchingMethod;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage() {
  /*while ((opt = getopt(argc, argv, "h:x:y:c:i:v:p:o:s:j:k:")) >= 0) {*/
  fprintf(
      stderr,
      "\nUsage:  ./spmv.out [options]"
      "\n"
      "\n    -l    input size (default=65536 elements)"
      "\n    -c    dramsim config file"
      "\n    -i    input file containing two vectors (default=generates vector "
      "with random numbers)"
      "\n    -x    number of columns (in blocks) in the generated test matrix"
      "\n    -y    number of rows (in blocks) in the generated test matrix"
      "\n    -s    seed of the generation (default 42)"
      "\n    -p    maximum number of populated blocks per row"
      "\n    -o    minimum number of populated (occupied) blocks per row"
      "\n    -v    t = verifies PIM output with host output. (default=false)"
      "\n    -h    shows help text"
      "\n");
}

struct Params getInputParams(int argc, char **argv) {
  struct Params p;
  // default generation settings
  p.matRowsBlocks = 1024;
  p.matColsBlocks = 1024;
  p.maxBlocksPerRow = 32;
  p.minBlocksPerRow = 0;
  p.maxNNZPerBlock = 32;
  p.minNNZPerBlock = 1;
  p.seed = 42;
  p.batchingMethod = 0;

  p.maxVal = 20;

  p.nnz = 0;

  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:x:y:c:i:v:p:o:s:j:k:n:b:")) >= 0) {
    switch (opt) {
    case 'h':
      usage();
      exit(0);
      break;
    case 'x':
      p.matColsBlocks = strtoull(optarg, NULL, 0);
      break;
    case 'y':
      p.matRowsBlocks = strtoull(optarg, NULL, 0);
      break;
    case 'c':
      p.configFile = optarg;
      break;
    case 'i':
      p.inputFile = optarg;
      break;
    case 'v':
      p.shouldVerify = (*optarg == 't') ? true : false;
      break;
    case 'p':
      p.maxBlocksPerRow = strtoull(optarg, NULL, 0);
      break;
    case 'o':
      p.minBlocksPerRow = strtoull(optarg, NULL, 0);
      break;
    case 's':
      p.seed = strtoull(optarg, NULL, 0);
      break;
    case 'j':
      p.maxNNZPerBlock = strtoull(optarg, NULL, 0);
      break;
    case 'k':
      p.minNNZPerBlock = strtoull(optarg, NULL, 0);
      break;
    case 'n':
      p.nnz = strtoull(optarg, NULL, 0);
      break;
    case 'b':
      p.batchingMethod = strtoull(optarg, NULL, 0);
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

// algorithm source:
// https://www.noweherenearithaca.com/2013/05/robert-floyds-tiny-and-beautiful.html
std::set<uint64_t> randomSet(uint64_t N, uint64_t M, std::mt19937 *gen) {
  std::set<uint64_t> S;
  std::pair<std::set<uint64_t>::iterator, bool> ret;
  for (uint64_t j = N - M; j < N; j++) {
    uint64_t t = std::uniform_int_distribution<uint64_t>(0, j)(*gen);
    ret = S.insert(t);
    if (!ret.second) {
      S.insert(j);
    }
  }
  return S;
}

void mergeWithKey(int32_t *values, uint64_t *keys, size_t left, size_t mid,
                  size_t right) {
  size_t mid_right = mid + 1;

  if (keys[mid] <= keys[mid_right]) {
    return;
  }

  while (left <= mid && mid_right <= right) {
    if (keys[left] <= keys[mid_right]) {
      left++;
    } else {
      uint64_t value = values[mid_right];
      uint64_t key = keys[mid_right];
      size_t index = mid_right;

      while (index != left) {
        keys[index] = keys[index - 1];
        values[index] = values[index - 1];
        index--;
      }
      values[left] = value;
      keys[left] = key;

      left++;
      mid++;
      mid_right++;
    }
  }
}

void mergeSortKey(int32_t *values, uint64_t *keys, size_t right,
                  size_t left = 0) {
  if (left < right) {
    size_t mid = left + (right - left) / 2;

    mergeSortKey(values, keys, mid, left);
    mergeSortKey(values, keys, left, mid + 1);

    mergeWithKey(values, keys, left, mid, right);
  }
}

void generateSpMV(struct Params p, std::vector<int32_t> *x,
                  std::vector<int32_t> *values,
                  std::vector<uint64_t> *rowPointers,
                  std::vector<uint64_t> *colPointers) {
  srand(p.seed);
  std::mt19937 gen(p.seed);

  /*bitmaps->resize(p.matRowsBlocks * p.matColsBlocks);*/
  /*offsets->resize(p.matRowsBlocks * p.matColsBlocks);*/

  if (p.nnz != 0) {
    values->resize(p.nnz);
    for (size_t i = 0; i < p.nnz; i++)
      (*values)[i] = (rand() % (p.maxVal - 1) + 1);

    x->resize(p.matColsBlocks * BLOCK_LEN);
    for (size_t i = 0; i < x->size(); i++)
      (*x)[i] = rand() % p.maxVal;
    std::set<uint64_t> positions = randomSet(
        p.matColsBlocks * p.matRowsBlocks * BLOCK_LEN * BLOCK_LEN, p.nnz, &gen);
    std::vector<uint64_t> elementPositions(positions.begin(), positions.end());

    for (size_t i = 0; i < elementPositions.size(); i++) {
      size_t blockPos = elementPositions[i] % (BLOCK_LEN * BLOCK_LEN);
      size_t blockRow =
          static_cast<size_t>(ceil((1.0f * blockPos) / (1.0f * BLOCK_LEN)));
      size_t blockCol = blockPos % BLOCK_LEN;
      size_t blockNum = static_cast<size_t>((1.0f * elementPositions[i]) /
                                            (1.0f * BLOCK_LEN * BLOCK_LEN));
      size_t row =
          static_cast<size_t>((1.0f * blockNum) / (1.0f * p.matColsBlocks));
      size_t col = blockNum % p.matColsBlocks;

      size_t realPosition = row * BLOCK_LEN * BLOCK_LEN * p.matColsBlocks +
                            blockRow * p.matColsBlocks * BLOCK_LEN +
                            col * BLOCK_LEN + blockCol;
      elementPositions[i] = realPosition;
    }

    mergeSortKey((int32_t *)values->data(), (uint64_t *)elementPositions.data(),
                 values->size() - 1);

    rowPointers->resize(p.matRowsBlocks * BLOCK_LEN + 1);
    colPointers->resize(p.nnz);

    uint64_t currentRow = 1;
    for (uint64_t i = 0; i < elementPositions.size(); i++) {
      uint64_t row = elementPositions[i] / (p.matColsBlocks * BLOCK_LEN);
      uint64_t col = elementPositions[i] % (p.matColsBlocks * BLOCK_LEN);
      if (row > currentRow) {
        while (currentRow < row) {
          (*rowPointers)[currentRow + 1] = (*rowPointers)[currentRow];
          currentRow++;
        }
      }
      (*colPointers)[i] = col;
      (*rowPointers)[currentRow]++;
    }

    if ((*rowPointers)[currentRow] > p.nnz) {
      for (uint64_t i = rowPointers->size(); i > currentRow; i--) {
        (*rowPointers)[i] = (*rowPointers)[currentRow];
      }
    }
    return;
  }
  std::cout << "Must provide a specified number of nonzero values."
            << std::endl;
  return;
}

void cpuCSRSpmv(const std::vector<int32_t> &values, size_t valuesLength,
                const std::vector<int32_t> &x, size_t xLength,
                const std::vector<uint64_t> &rowPointers,
                size_t rowPointersLength,
                const std::vector<uint64_t> &colPointers,
                size_t colPointersLength, std::vector<int32_t> &dst) {
  dst.resize(rowPointersLength - 1);
  std::fill(dst.begin(), dst.end(), 0);

  for (uint64_t i = 0; i < rowPointersLength - 1; i++) {
    for (uint64_t j = rowPointers[i]; j < rowPointers[i + 1]; j++) {
      dst[i] += x[colPointers[j]] * values[j];
    }
  }
}

int main(int argc, char *argv[]) {
  struct Params params = getInputParams(argc, argv);
  // std::cout << "Vector length: " << params.vectorLength << "\n";
  std::vector<int32_t> x, values, Y, dst;
  std::vector<uint64_t> matDim, rowPointers, colPointers;
  // std::vector<int> src1, src2, dst;
  if (params.inputFile == nullptr) {
    generateSpMV(params, &x, &values, &rowPointers, &colPointers);
  } else {
    std::ifstream BSR_mat_file(params.inputFile);
    std::string buffer;
    int flen;
    if (BSR_mat_file.is_open()) {
      BSR_mat_file.seekg(0, BSR_mat_file.end);
      // find length of file
      flen = BSR_mat_file.tellg();
      BSR_mat_file.seekg(0, BSR_mat_file.beg);
      // initialize the buffer
      buffer = std::string(flen, ' ');
      // read the file
      BSR_mat_file.read(&buffer[0], flen);
    }

    // read the buffer for parameters
    if (buffer.length() > 0) {
      readLineToVec(buffer, x);
      readLineToVec(buffer, values);
      if (params.shouldVerify) {
        readLineToVec(buffer, Y);
      }
    }
  }

  auto start = std::chrono::high_resolution_clock::now();
  const int BENCH_NUM = 2;

  for (int i = 0; i < BENCH_NUM; i++)
    cpuCSRSpmv(values, values.size(), x, x.size(), rowPointers,
               rowPointers.size(), colPointers, colPointers.size(), dst);

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsedTime =
      (end - start) / BENCH_NUM;

  std::cout << "Average duration: " << std::fixed << std::setprecision(3)
            << elapsedTime.count() << " ms." << std::endl;

  std::vector<int32_t> xRearranged(values.size());

  start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < BENCH_NUM; i++)
    for (uint64_t i = 0; i < colPointers.size(); i++) {
      /*status = pimCopyHostToDevice((void *)&x.data()[colPointers[i]], xDevice,
       * i,*/
      /*                             i + 1);*/
      /*assert(status == PIM_OK);*/
      xRearranged[i] = x[colPointers[i]];
    }

  end = std::chrono::high_resolution_clock::now();

  elapsedTime = (end - start) / BENCH_NUM;

  std::cout << "Rearrangement Duration: " << std::fixed << std::setprecision(3)
            << elapsedTime.count() << " ms." << std::endl;

  return 0;
}
