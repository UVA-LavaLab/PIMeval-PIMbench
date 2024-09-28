// Test: C++ version of the game of life
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#include <cassert>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../../util.h"
#include "libpimeval.h"

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t width;
  uint64_t height;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./game-of-life.out [options]"
          "\n"
          "\n    -x    board width (default=2048 elements)"
          "\n    -y    board height (default=2048 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing a game board (default=generates board with random states)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.width = 2048;
  p.height = 2048;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:x:y:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'x':
      p.width = strtoull(optarg, NULL, 0);
      break;
    case 'y':
      p.height = strtoull(optarg, NULL, 0);
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
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

void print_pim_obj(PimObjId pim_obj, size_t sz) {
  std::vector<uint8_t> tmp_res;
  tmp_res.resize(sz);
  pimCopyDeviceToHost(pim_obj, tmp_res.data());

  for(size_t i=0; i<sz; ++i) {
    std::cout << unsigned(tmp_res[i]) << ", ";
  }
  std::cout << std::endl;
}

void game_of_life_row(const std::vector<PimObjId> &pim_board, size_t row_idx, PimObjId tmp_pim_obj, const std::vector<PimObjId>& pim_sums, int old_ind, PimObjId result_obj) {
  size_t mid_idx = 3*row_idx + 1;

  pimAdd(pim_board[mid_idx + 2], pim_board[mid_idx + 3], pim_sums[old_ind]);
  pimAdd(pim_board[mid_idx + 4], pim_sums[old_ind], pim_sums[old_ind]);

  pimAdd(pim_sums[old_ind], pim_sums[(old_ind + 1) % pim_sums.size()], tmp_pim_obj);
  pimAdd(pim_board[mid_idx - 1], tmp_pim_obj, tmp_pim_obj);
  pimAdd(pim_board[mid_idx + 1], tmp_pim_obj, tmp_pim_obj);
  
  unsigned bitsPerElement = 8;

  PimStatus status = pimEQScalar(tmp_pim_obj, result_obj, 3);
  assert (status == PIM_OK);

  status = pimEQScalar(tmp_pim_obj, tmp_pim_obj, 2);
  assert (status == PIM_OK);

  status = pimAnd(tmp_pim_obj, pim_board[mid_idx], tmp_pim_obj);
  assert (status == PIM_OK);

  status = pimOr(tmp_pim_obj, result_obj, result_obj);
  assert (status == PIM_OK);
}

void add_vector_to_grid(const std::vector<uint8_t> &to_add, PimObjId to_associate, std::vector<PimObjId> &pim_board) {

  unsigned bitsPerElement = 8;

  PimObjId mid = pimAllocAssociated(bitsPerElement, to_associate, PIM_UINT8);
  assert(mid != -1);
  PimStatus status = pimCopyHostToDevice((void *)to_add.data(), mid);
  assert (status == PIM_OK);

  PimObjId left = pimAllocAssociated(bitsPerElement, mid, PIM_UINT8);
  assert(left != -1);
  status = pimCopyDeviceToDevice(mid, left);
  assert (status == PIM_OK);
  status = pimShiftElementsRight(left);
  assert (status == PIM_OK);

  

  PimObjId right = pimAllocAssociated(bitsPerElement, mid, PIM_UINT8);
  assert(right != -1);
  status = pimCopyDeviceToDevice(mid, right);
  assert (status == PIM_OK);
  status = pimShiftElementsLeft(right);
  assert (status == PIM_OK);

  pim_board.push_back(left);
  pim_board.push_back(mid);
  pim_board.push_back(right);
}

// Will alloc:
// 1: tmp_pim_obj

// Add vector to grid:
// 2 calls
// end_y-start_y calls
// 3 alloc per call
// Total: 3*(2+end_y-start_y) alloc assoc calls

// 3 tmp sum objs

// gol row
// 1 alloc
// runs end_y-start_y times

// Total:
// 1: tmp pim obj
// 3*(2+end_y-start_y): add vec to grid
// 3: tmp smp
// (end_y-start_y): results

// 1+3+3*(2+end_y-start_y)+(end_y-start_y)
// 10+4*(end_y-start_y)

// New with optim:
// 11+3(end_y-start_y)


void game_of_life(const std::vector<std::vector<uint8_t>> &src_host, std::vector<std::vector<uint8_t>> &dst_host,
size_t start_x, size_t end_x, size_t start_y, size_t end_y)
{
  unsigned bitsPerElement = 8;
  size_t width = end_x - start_x;
  size_t height = end_y - start_y;

  PimObjId tmp_pim_obj = pimAlloc(PIM_ALLOC_AUTO, width, bitsPerElement, PIM_UINT8);
  assert(tmp_pim_obj != -1);

  std::vector<PimObjId> pim_board;

  if(start_y > 0) {
    add_vector_to_grid(src_host[start_y - 1], tmp_pim_obj, pim_board);
  } else {
    std::vector<uint8_t> tmp_zeros(width, 0);
    add_vector_to_grid(tmp_zeros, tmp_pim_obj, pim_board);
  }

  for(size_t i=start_y; i<end_y; ++i) {
    add_vector_to_grid(src_host[i], tmp_pim_obj, pim_board);
  }

  if(end_y < src_host.size()) {
    add_vector_to_grid(src_host[end_y], tmp_pim_obj, pim_board);
  } else {
    std::vector<uint8_t> tmp_zeros(width, 0);
    add_vector_to_grid(tmp_zeros, tmp_pim_obj, pim_board);
  }

  std::vector<PimObjId> tmp_sums;

  tmp_sums.push_back(pimAllocAssociated(bitsPerElement, tmp_pim_obj, PIM_UINT8));
  tmp_sums.push_back(pimAllocAssociated(bitsPerElement, tmp_pim_obj, PIM_UINT8));
  tmp_sums.push_back(pimAllocAssociated(bitsPerElement, tmp_pim_obj, PIM_UINT8));

  PimStatus status = pimAdd(pim_board[0], pim_board[1], tmp_sums[0]);
  assert (status == PIM_OK);
  status = pimAdd(pim_board[2], tmp_sums[0], tmp_sums[0]);
  assert (status == PIM_OK);

  status = pimAdd(pim_board[3], pim_board[4], tmp_sums[1]);
  assert (status == PIM_OK);
  status = pimAdd(pim_board[5], tmp_sums[1], tmp_sums[1]);
  assert (status == PIM_OK);

  int old_ind = 2;

  PimObjId result_object = pimAllocAssociated(bitsPerElement, tmp_pim_obj, PIM_UINT8);
  assert(result_object != -1);

  for(size_t i=1; i<height+1; ++i) {
    game_of_life_row(pim_board, i, tmp_pim_obj, tmp_sums, old_ind, result_object);
    old_ind = (1+old_ind) % tmp_sums.size();
    PimStatus copy_status = pimCopyDeviceToHost(result_object, dst_host[start_y + i - 1].data());
    assert (copy_status == PIM_OK);
  }

  pimFree(tmp_pim_obj);

  for(size_t i=0; i<pim_board.size(); ++i) {
    pimFree(pim_board[i]);
  }

  pimFree(result_object);

  for(size_t i=0; i<tmp_sums.size(); ++i) {
    pimFree(tmp_sums[i]);
  }
}

uint8_t get_with_default(size_t i, size_t j, std::vector<std::vector<uint8_t>> &x) {
  if(i >= 0 && i < x.size() && j >= 0 && j < x[0].size()) {
    return x[i][j];
  }
  return 0;
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Running PIM game of life for board: " << params.width << "x" << params.height << "\n";
  std::vector<std::vector<uint8_t>> x, y;
  if (params.inputFile == nullptr)
  {
    srand((unsigned)time(NULL));
    x.resize(params.height);
    for(size_t i=0; i<params.height; ++i) {
      x[i].resize(params.width);
      for(size_t j=0; j<params.width; ++j) {
        x[i][j] = rand() & 1;
      }
    }
  } 
  else 
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 1;
  }
  
  if (!createDevice(params.configFile))
  {
    return 1;
  }

  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  assert(status == PIM_OK);
  uint64_t numCol = deviceProp.numColPerSubarray, numRow = deviceProp.numRowPerSubarray, 
           numCore = deviceProp.numRanks * deviceProp.numBankPerRank * deviceProp.numSubarrayPerBank;
  uint64_t totalAvailableBits = numCol * numRow * numCore;

  std::cout << "rows: " << numRow << std::endl;
  std::cout << "total bits: " << totalAvailableBits << std::endl;

  // TODO: Fix
  bool is_v_device = true;

  // TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iteration.
  y.resize(x.size());
  for(size_t i=0; i<y.size(); ++i) {
    y[i].resize(x[0].size());
  }

  unsigned bitsPerElement = 8;
  int max_alloc_assoc;
  if(is_v_device) {
    max_alloc_assoc = numRow/bitsPerElement;
  } else {
    max_alloc_assoc = numRow;
  }

  // Dont understand this line, seems to work though, because if stride incremented any more, then it fails
  // Not accurate for bank level
  max_alloc_assoc *= 2;

  // stride = end_y-start_y
  // 11+3*stride
  // max_alloc = 11+3*stride
  // (max_alloc-11)/3 = stride

  int stride = (max_alloc_assoc-11)/3;

  for(size_t i=0; i<params.height; i += stride) {
    game_of_life(x, y, 0, params.width, i, min(i+stride, params.height));
  }

  if (params.shouldVerify) 
  {
    bool is_correct = true;
    for(int i=0; i<y.size(); ++i) {
      for(int j=0; j<y[0].size(); ++j) {
        uint8_t sum_cpu = get_with_default(i-1, j-1, x);
        sum_cpu += get_with_default(i-1, j, x);
        sum_cpu += get_with_default(i-1, j+1, x);
        sum_cpu += get_with_default(i, j-1, x);
        sum_cpu += get_with_default(i, j+1, x);
        sum_cpu += get_with_default(i+1, j-1, x);
        sum_cpu += get_with_default(i+1, j, x);
        sum_cpu += get_with_default(i+1, j+1, x);

        uint8_t res_cpu = (sum_cpu == 3) ? 1 : 0;
        sum_cpu = (sum_cpu == 2) ? 1 : 0;
        sum_cpu &= get_with_default(i, j, x);
        res_cpu |= sum_cpu;

        if (res_cpu != y[i][j])
        {
          std::cout << "Wrong answer: " << unsigned(y[i][j]) << " (expected " << unsigned(res_cpu) << ")" << std::endl;
          is_correct = false;
        }
      }
    }
    if(is_correct) {
      std::cout << "Correct for game of life!" << std::endl;
    }
  }

  pimShowStats();

  return 0;
}