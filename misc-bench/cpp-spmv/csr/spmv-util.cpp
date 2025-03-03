#include "spmv-util.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../../../libpimeval/src/libpimeval.h"

struct Params getInputParams(int argc, char **argv) {
  struct Params p;
  // default generation settings
  p.matRowsBlocks = 1024;
  p.matColsBlocks = 1024;
  p.seed = 42;

  p.maxVal = 20;

  p.nnz = 0;

  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;
  p.deviceType = PIM_FUNCTIONAL;

  int opt;
  while ((opt = getopt(argc, argv, "h:x:y:c:i:v:s:n:d:g:")) >= 0) {
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
    case 's':
      p.seed = strtoull(optarg, NULL, 0);
      break;
    case 'n':
      p.nnz = strtoull(optarg, NULL, 0);
      break;
    case 'd':
      if (optarg[0] == 'f') {
        p.deviceType = PIM_DEVICE_FULCRUM;
      } else if (optarg[0] == 'b') {
        p.deviceType = PIM_DEVICE_BANK_LEVEL;
      }
      break;
    case 'g':
      p.generationType = GEN_RANDOM_BIASED;
      if (optarg[0] == 'u') {
        p.generationType = GEN_UNIFORM;
      } else if (optarg[0] == 'l') {
        p.generationType = GEN_LEFT_ALIGNED;
      } else if (optarg[0] == 't') {
        p.generationType = GEN_TRIDIAGONAL;
      }
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

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

void cpuCSRSpmv(const std::vector<float> &values, size_t valuesLength,
                const std::vector<float> &x, size_t xLength,
                const std::vector<u64> &rowPointers, size_t rowPointersLength,
                const std::vector<u64> &colPointers, size_t colPointersLength,
                std::vector<float> &dst) {
  dst.resize(rowPointersLength - 1);
  std::fill(dst.begin(), dst.end(), 0);

  for (u64 i = 0; i < rowPointersLength - 1; i++) {
    for (u64 j = rowPointers[i]; j < rowPointers[i + 1]; j++) {
      dst[i] += x[colPointers[j]] * values[j];
    }
  }
}

// TODO: refactor for float
void generateUniformSpMV(struct Params *p, std::vector<float> *x,
                         std::vector<float> *values,
                         std::vector<u64> *rowPointers,
                         std::vector<u64> *colIndex) {
  u64 numRow = p->matRowsBlocks * BLOCK_LEN;
  u64 numCol = p->matColsBlocks * BLOCK_LEN;
  u64 nnz = p->nnz;
  double percentage = (float)nnz / (numRow * numCol);
  int seed = p->seed;

  rowPointers->resize(numRow + 1);
  colIndex->resize(nnz);
  values->resize(nnz);
  x->resize(numCol);

  (*rowPointers)[0] = 0;
  srand(seed);
  u64 nzPerRow = 1u + (u64)((1.0f * numCol) * percentage);
  u64 actualNonzeros = 0;

  std::mt19937 gen(seed);
  std::vector<u64> nonzeroLocations(numCol);
  std::iota(nonzeroLocations.begin(), nonzeroLocations.end(), 1u);
  std::vector<u64> rowPositions;
  for (u64 i = 0; i < numRow; i++) {
    (*rowPointers)[i] = actualNonzeros;
    u64 tIndex = 0;
    std::sample(nonzeroLocations.begin(), nonzeroLocations.end(),
                std::back_inserter(rowPositions), nzPerRow, gen);

    for (u64 j = 0; j < nzPerRow; j++) {
      u64 temp = rand() % (numCol / nzPerRow) + 1;
      tIndex += temp;
      if (actualNonzeros < nnz) {
        (*values)[actualNonzeros] = (float)rand() / RAND_MAX;
        (*colIndex)[actualNonzeros] = rowPositions[j];
        actualNonzeros++;
      }
    }
    rowPositions.resize(0);
  }

  (*rowPointers)[numRow] = nnz;
  if (nnz != actualNonzeros) {
    std::cout << "Error, number of nonzeros does not equal the actual number "
                 "of nonzeros"
              << std::endl;
    exit(1);
  }
  for (u64 i = 0; i < numCol; i++)
    (*x)[i] = (float)rand() / RAND_MAX;
}

void generateSpMV(struct Params *p, std::vector<float> *x,
                  std::vector<float> *values, std::vector<u64> *rowPointers,
                  std::vector<u64> *colPointers) {
  srand(p->seed);
  std::mt19937 gen(p->seed);

  /*bitmaps->resize(p.matRowsBlocks * p.matColsBlocks);*/
  /*offsets->resize(p.matRowsBlocks * p.matColsBlocks);*/

  if (p->nnz != 0) {
    values->resize(p->nnz);
    for (size_t i = 0; i < p->nnz; i++)
      (*values)[i] = ((float)rand() / RAND_MAX) + 10.0f;

    x->resize(p->matColsBlocks * BLOCK_LEN);
    for (size_t i = 0; i < x->size(); i++)
      (*x)[i] = (float)rand() / RAND_MAX;
    std::set<u64> positions =
        randomSet(p->matColsBlocks * p->matRowsBlocks * BLOCK_LEN * BLOCK_LEN,
                  p->nnz, &gen);
    std::vector<u64> elementPositions(positions.begin(), positions.end());

    std::cout << "Element positions size: " << elementPositions.size()
              << std::endl;
    for (size_t i = 0; i < elementPositions.size(); i++) {
      size_t blockPos = elementPositions[i] % (BLOCK_LEN * BLOCK_LEN);
      size_t blockRow =
          static_cast<size_t>(ceil((1.0f * blockPos) / (1.0f * BLOCK_LEN)));
      size_t blockCol = blockPos % BLOCK_LEN;
      size_t blockNum = static_cast<size_t>((1.0f * elementPositions[i]) /
                                            (1.0f * BLOCK_LEN * BLOCK_LEN));
      size_t row =
          static_cast<size_t>((1.0f * blockNum) / (1.0f * p->matColsBlocks));
      size_t col = blockNum % p->matColsBlocks;

      size_t realPosition = row * BLOCK_LEN * BLOCK_LEN * p->matColsBlocks +
                            blockRow * p->matColsBlocks * BLOCK_LEN +
                            col * BLOCK_LEN + blockCol;
      elementPositions[i] = realPosition;
    }

    mergeSortKey<float>((float *)values->data(), (u64 *)elementPositions.data(),
                        values->size() - 1);

    rowPointers->resize(p->matRowsBlocks * BLOCK_LEN + 1);
    colPointers->resize(p->nnz);

    u64 currentRow = 1;
    for (u64 i = 0; i < elementPositions.size(); i++) {
      u64 row = elementPositions[i] / (p->matColsBlocks * BLOCK_LEN);
      u64 col = elementPositions[i] % (p->matColsBlocks * BLOCK_LEN);
      if (row > currentRow) {
        while (currentRow < row) {
          (*rowPointers)[currentRow + 1] = (*rowPointers)[currentRow];
          currentRow++;
        }
      }
      (*colPointers)[i] = col;
      (*rowPointers)[currentRow]++;
    }

    if ((*rowPointers)[currentRow] >= p->nnz) {
      for (u64 i = rowPointers->size() - 1; i > currentRow; i--) {
        (*rowPointers)[i] = (*rowPointers)[currentRow];
      }
    }
    return;
  }
  std::cout << "Must provide a specified number of nonzero values."
            << std::endl;
  return;
}

// algorithm source:
// https://www.noweherenearithaca.com/2013/05/robert-floyds-tiny-and-beautiful.html
std::set<u64> randomSet(u64 N, u64 M, std::mt19937 *gen) {
  std::set<u64> S;
  std::pair<std::set<u64>::iterator, bool> ret;
  for (u64 j = N - M + 1; j < N; j++) {
    u64 t = std::uniform_int_distribution<u64>(0, j)(*gen);
    ret = S.insert(t);
    if (!ret.second) {
      S.insert(j);
    }
  }
  return S;
}

// only needs numRow because numRow = numCol
/*void generateSpMV(struct Params p, std::vector<float> *x,*/
/*                  std::vector<float> *values, std::vector<u64> *rowPointers,*/
/*                  std::vector<u64> *colPointers) {*/
void generateTriDiagonal(struct Params *p, std::vector<float> *x,
                         std::vector<float> *values,
                         std::vector<u64> *rowPointers,
                         std::vector<u64> *colIndex) {
  u64 numRow = p->matRowsBlocks * BLOCK_LEN;
  u64 *nnz = &p->nnz;

  if (numRow == 1u) {
    (*nnz) = 1u;
  } else if (numRow >= 2u) {
    (*nnz) = 4u;
    (*nnz) += 3u * (numRow - 2);
  }
  rowPointers->resize(numRow + 1);
  colIndex->resize(*nnz);
  values->resize(*nnz);
  x->resize(numRow);

  (*rowPointers)[0] = 0, (*colIndex)[0] = 0, (*colIndex)[1] = 1;
  (*values)[0] = (float)rand() / RAND_MAX;
  (*values)[1] = (float)rand() / RAND_MAX;
  int start;

  for (u64 i = 1; i < numRow; i++) {
    if (i > 1) {
      (*rowPointers)[i] = (*rowPointers)[i - 1] + 3;
    } else {
      (*rowPointers)[1] = 2;
    }

    start = (i - 1) * 3 + 2;
    (*colIndex)[start] = i - 1;
    (*colIndex)[start + 1] = i;

    if (i < numRow - 1) {
      (*colIndex)[start + 2] = i + 1;
    }

    (*values)[start] = (*values)[start - 1];
    (*values)[start + 1] = (float)rand() / RAND_MAX;

    if (i < numRow - 1) {
      (*values)[start + 2] = (float)rand() / RAND_MAX;
    }
  }
  (*rowPointers)[numRow] = *nnz;

  for (u64 i = 0; i < numRow; i++)
    (*x)[i] = (float)rand() / RAND_MAX;
}

void generateRandomNonzero(struct Params *p, std::vector<float> *x,
                           std::vector<float> *values,
                           std::vector<u64> *rowPointers,
                           std::vector<u64> *colIndex) {
  u64 numRow = p->matRowsBlocks * BLOCK_LEN;
  u64 numCol = p->matColsBlocks * BLOCK_LEN;
  u64 nnz = p->nnz;
  double percentage = (float)nnz / (numRow * numCol);

  rowPointers->resize(numRow + 1);
  colIndex->resize(nnz);
  values->resize(nnz);
  x->resize(numRow);

  (*rowPointers)[0] = 0, (*colIndex)[0] = 0;
  srand(p->seed);
  u64 nzPerRow = 1u + (u64)((1.0f * numCol) * percentage);
  if (nzPerRow > numCol) {
    fprintf(stderr, "Error: too many nonzeros for this matrix size");
    exit(1);
  }
  u64 actualNonzeros = 0;

  for (u64 i = 0; i < numRow; i++) {
    (*rowPointers)[i] = actualNonzeros;
    for (u64 j = 0; j < nzPerRow; j++) {
      /*u64 temp = rand() % (numCol / nzPerRow) + 1;*/
      if (actualNonzeros < nnz) {
        (*values)[actualNonzeros] = (float)rand() / RAND_MAX;
        (*colIndex)[actualNonzeros] = j;
        actualNonzeros++;
      }
    }
  }

  (*rowPointers)[numRow] = nnz;
  if (nnz != actualNonzeros) {
    std::cout << "Error, number of nonzeros does not equal the actual number "
                 "of nonzeros"
              << std::endl;
    exit(1);
  }

  for (u64 i = 0; i < numCol; i++)
    (*x)[i] = (float)rand() / RAND_MAX;
}

void readMatrixFromFile(char *sparseFilename, std::vector<u64> *rowPointers,
                        std::vector<u64> *colIndex, std::vector<float> *values,
                        std::vector<float> *x, u64 seed) {
  std::string line, entry;
  std::ifstream sparseFile(sparseFilename);
  u64 nnz = 0, numRow = 0, numCol = 0;
  struct {
    u64 row, col;
    float value;
  } buffer = {0u, 0u, 0.0f};

  if (sparseFile.is_open()) {
    // remove the comments at the top of the file
    while (std::getline(sparseFile, line) && line[0] == '%') {
    }
    std::stringstream s1(line);
    // the third (and last) element in the first non-comment row is the length
    // of the file
    for (u64 i = 0; i < 3; i++) {
      std::getline(s1, entry, ' ');
      switch (i) {
      case 0:
        numRow = std::stoull(entry);
        break;
      case 1:
        numCol = std::stoull(entry);
        break;
      default:
        nnz = std::stoull(entry);
      }
    }
    if (numRow == 0 or numCol == 0) {
      fprintf(stderr, "Could not parse matrix dimensions.");
      exit(1);
    }

    rowPointers->resize(numRow);
    colIndex->resize(nnz);
    values->resize(nnz);
    x->resize(numCol);

    std::vector<Point> points(nnz);
    std::vector<u64> rowPos(nnz);

    for (u64 i = 0; i < nnz; i++) {
      std::getline(sparseFile, line);

      bool valFound = false;
      std::stringstream s1(line);
      for (u64 j = 0; j < 3; j++) {
        if (std::getline(s1, entry, ' ')) {
          switch (j) {
          case 0:
            buffer.row = std::stoull(entry);
            break;
          case 1:
            buffer.col = std::stoull(entry);
            break;
          default:
            if (entry.size() > 0) {
              buffer.value = std::atof(entry.c_str());
              valFound = true;
            }
          }
        }
      }

      if (valFound) {
        points[i] = Point{buffer.row, buffer.col, buffer.value};
      } else {
        points[i] =
            Point{buffer.row, buffer.col, (float)rand() / RAND_MAX + 10.0f};
      }
      rowPos[i] = buffer.row;
    }

    std::sort(points.begin(), points.end(), [](const Point &a, const Point &b) {
      if (a.x == b.x) {
        return a.y < b.y;
      }
      return a.x < b.x;
    });

    for (const auto &p : points) {
      (*rowPointers)[p.x]++;
    }

    for (u64 i = 1; i < numRow + 1; i++) {
      (*rowPointers)[i] += (*rowPointers)[i - 1];
    }

    u64 pos = 0;
    for (const auto &p : points) {
      (*colIndex)[pos] = p.y - 1;
      (*values)[pos] = p.z;
      pos++;
    }
  } else {
    std::cout << "File could not be opened." << std::endl;
    exit(1);
  }
}
