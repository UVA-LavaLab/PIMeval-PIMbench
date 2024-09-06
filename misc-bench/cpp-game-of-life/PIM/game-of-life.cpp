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

PimObjId game_of_life_row(const std::vector<PimObjId> &pim_board, size_t row_idx, PimObjId tmp_pim_obj, const std::vector<PimObjId>& pim_sums, int old_ind) {

  // 1
  // 2
  // 3
  // 4
  // 5
  // 6
  // 7
  // 8
  // 9
  // 10
  // 11
  // 12
  // 13
  // 14
  // 15


  // 123 -> A
  // 456 -> B
  // 789 -> C
  // A+B+C - 5
  // 10,11,12 -> A
  // A+B+C - 8
  // Transition Price:
  // 2 adds to set A/B/C (e.g. (1+2)+3)
  // res = A+B+C-5, 3 adds
  // transition is 5 adds


  // Add first 9
  // 1,2,3,4,6,7,8,9

  // -1, -2, -3, +5, +10, +11, +12
  // 4,5,6,7,9,10,11,12

  // -4, -5, -6, +8, +13, +14, +15
  // 7,8,9,10,12,13,14,15

  // first to third:
  // -1, -2, -3, -4, -6, + (10,12,13,14,15)

  size_t mid_idx = 3*row_idx + 1;

  pimAdd(pim_board[mid_idx + 2], pim_board[mid_idx + 3], pim_sums[old_ind]);
  pimAdd(pim_board[mid_idx + 4], pim_sums[old_ind], pim_sums[old_ind]);

  pimAdd(pim_sums[old_ind], pim_sums[(old_ind + 1) % pim_sums.size()], tmp_pim_obj);
  pimAdd(pim_board[mid_idx - 1], tmp_pim_obj, tmp_pim_obj);
  pimAdd(pim_board[mid_idx + 1], tmp_pim_obj, tmp_pim_obj);

  // pimAdd(pim_board[mid_idx - 1], pim_board[mid_idx + 1], tmp_pim_obj);
  // pimAdd(pim_board[mid_idx - 2], tmp_pim_obj, tmp_pim_obj);
  // pimAdd(pim_board[mid_idx - 3], tmp_pim_obj, tmp_pim_obj);
  // pimAdd(pim_board[mid_idx - 4], tmp_pim_obj, tmp_pim_obj);

  // pimAdd(pim_board[mid_idx + 2], tmp_pim_obj, tmp_pim_obj);
  // pimAdd(pim_board[mid_idx + 3], tmp_pim_obj, tmp_pim_obj);
  // pimAdd(pim_board[mid_idx + 4], tmp_pim_obj, tmp_pim_obj);
  
  unsigned bitsPerElement = 8;
  PimObjId pim_res = pimAllocAssociated(bitsPerElement, pim_board[mid_idx], PIM_UINT8);
  assert(pim_res != -1);

  PimStatus status = pimEQScalar(tmp_pim_obj, pim_res, 3);
  assert (status == PIM_OK);

  status = pimEQScalar(tmp_pim_obj, tmp_pim_obj, 2);
  assert (status == PIM_OK);

  status = pimAnd(tmp_pim_obj, pim_board[mid_idx], tmp_pim_obj);
  assert (status == PIM_OK);

  status = pimOr(tmp_pim_obj, pim_res, pim_res);
  assert (status == PIM_OK);

  return pim_res;
}

void add_vector_to_grid(const std::vector<uint8_t> &to_add, PimObjId to_associate, std::vector<PimObjId> &pim_board) {
  // Should be able to use a ranged ref to replace shift once implemented

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

// For a board of size x by y
// Allocates 1 tmp obj of length x
// Allocates 3*y objects of length x as input
// Allocates y objects of length x as output
// Total: 4y + 1 objects of length x
// uses (4y + 1)x * 8 bits
// Uses 4y + 1 rows of associated

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

  // std::cout << "start pim board: \n";
  
  // for(size_t i=0; i<pim_board.size(); ++i) {
  //   print_pim_obj(pim_board[i], width);
  // }

  // std::cout << "end pim board: \n";

  std::vector<PimObjId> result_objs;

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

  for(size_t i=1; i<height+1; ++i) {
    result_objs.push_back(game_of_life_row(pim_board, i, tmp_pim_obj, tmp_sums, old_ind));
    old_ind = (1+old_ind) % tmp_sums.size();
  }

  for(size_t i=0; i<height; ++i) {
    PimStatus copy_status = pimCopyDeviceToHost(result_objs[i], dst_host[start_y + i].data());
    assert (copy_status == PIM_OK);
  }

  pimFree(tmp_pim_obj);

  for(size_t i=0; i<pim_board.size(); ++i) {
    pimFree(pim_board[i]);
  }

  for(size_t i=0; i<result_objs.size(); ++i) {
    pimFree(result_objs[i]);
  }

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

    // x = {{1,0,1,0,0},
    //      {0,1,1,1,0},
    //      {0,0,0,1,1},
    //      {1,1,0,1,0},
    //      {1,0,0,1,1}};
         //0, 1, 0, 0, 1, 
         // 1, 0, 0, 0, 1,

          // Correct Board
          // 0, 0, 1, 1, 0, 
          // 0, 1, 0, 0, 1, 
          // 1, 0, 0, 0, 1, 
          // 1, 1, 0, 0, 0, 
          // 1, 1, 1, 1, 1
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

  // size_t i = 1;
  // PimObjId pim_obj = pimAlloc(PIM_ALLOC_AUTO, 1, 8, PIM_UINT8);

  // while(true) {
  //   PimObjId new_pim_obj = pimAllocAssociated(8, pim_obj, PIM_UINT8);
  //   if(new_pim_obj == -1) {
  //     std::cout << "allocated " << i << std::endl;
  //     exit(0);
  //   }
  //   ++i;
  // }

  // PimObjId test_obj = pimAlloc(PIM_ALLOC_AUTO, (totalAvailableBits / 8), 8, PIM_UINT8);
  // assert(test_obj != -1);

  // TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iteration.
  y.resize(x.size());
  for(size_t i=0; i<y.size(); ++i) {
    y[i].resize(x[0].size());
  }

  for(size_t i=0; i<params.height; i += 60) {
    game_of_life(x, y, 0, params.width, i, min(i+60, params.height));
  }
  

  // std::cout << "\n\nx: \n";

  // for(size_t i=0; i<x.size(); ++i) {
  //   for(size_t j=0; j<x[0].size(); ++j) {
  //     std::cout << unsigned(x[i][j]) << ", ";
  //   }
  //   std::cout << std::endl;
  // }

  // std::cout << "\n\ny: \n";

  // for(size_t i=0; i<y.size(); ++i) {
  //   for(size_t j=0; j<y[0].size(); ++j) {
  //     std::cout << unsigned(y[i][j]) << ", ";
  //   }
  //   std::cout << std::endl;
  // }

  // std::vector<std::vector<uint8_t>> y2;
  // std::cout << "\n\ny2: \n";
  // y2.resize(x.size());
  // for(size_t i=0; i<y2.size(); ++i) {
  //   y2[i].resize(x[0].size());
  // }
  // game_of_life(x, y2, 0, 5, 1, 3);
  // for(size_t i=0; i<y2.size(); ++i) {
  //   for(size_t j=0; j<y2[0].size(); ++j) {
  //     std::cout << unsigned(y2[i][j]) << ", ";
  //   }
  //   std::cout << std::endl;
  // }
  


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