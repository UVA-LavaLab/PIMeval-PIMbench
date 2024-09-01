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

PimObjId game_of_life_row(const std::vector<PimObjId> &pim_board, size_t row_idx, PimObjId tmp_pim_obj) {
  size_t mid_idx = 3*row_idx + 1;

  pimAdd(pim_board[mid_idx - 1], pim_board[mid_idx + 1], tmp_pim_obj);
  if(row_idx > 0) {
    pimAdd(pim_board[mid_idx - 2], tmp_pim_obj, tmp_pim_obj);
    pimAdd(pim_board[mid_idx - 3], tmp_pim_obj, tmp_pim_obj);
    pimAdd(pim_board[mid_idx - 4], tmp_pim_obj, tmp_pim_obj);
  }

  if(mid_idx + 2 < pim_board.size()) {
    pimAdd(pim_board[mid_idx + 2], tmp_pim_obj, tmp_pim_obj);
    pimAdd(pim_board[mid_idx + 3], tmp_pim_obj, tmp_pim_obj);
    pimAdd(pim_board[mid_idx + 4], tmp_pim_obj, tmp_pim_obj);
  }
  
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
  unsigned bitsPerElement = 8;

  PimObjId left = pimAllocAssociated(bitsPerElement, to_associate, PIM_UINT8);
  assert(left != -1);
  // Should be able to use a ranged ref to replace shift once implemented
  // Also try device to device copy
  PimStatus status = pimCopyHostToDevice((void *)to_add.data(), left);
  assert (status == PIM_OK);
  status = pimShiftElementsRight(left);
  assert (status == PIM_OK);

  PimObjId mid = pimAllocAssociated(bitsPerElement, to_associate, PIM_UINT8);
  assert(mid != -1);
  status = pimCopyHostToDevice((void *)to_add.data(), mid);
  assert (status == PIM_OK);

  PimObjId right = pimAllocAssociated(bitsPerElement, to_associate, PIM_UINT8);
  assert(right != -1);
  status = pimCopyHostToDevice((void *)to_add.data(), right);
  assert (status == PIM_OK);
  status = pimShiftElementsLeft(right);
  assert (status == PIM_OK);

  pim_board.push_back(left);
  pim_board.push_back(mid);
  pim_board.push_back(right);
}

void game_of_life(const std::vector<std::vector<uint8_t>> &src_host, std::vector<std::vector<uint8_t>> &dst_host)
{
  unsigned bitsPerElement = 8;
  size_t width = src_host[0].size();
  size_t height = src_host.size();

  PimObjId tmp_pim_obj = pimAlloc(PIM_ALLOC_AUTO, width, bitsPerElement, PIM_UINT8);
  assert(tmp_pim_obj != -1);

  std::vector<PimObjId> pim_board;

  for(size_t i=0; i<src_host.size(); ++i) {
    add_vector_to_grid(src_host[i], tmp_pim_obj, pim_board);
  }

  std::cout << "pim grid: \n";

  for(size_t i=0; i<pim_board.size(); ++i) {
    print_pim_obj(pim_board[i], width);
  }

  std::cout << "end pim grid\n";

  std::vector<PimObjId> result_objs;

  for(size_t i=0; i<src_host.size(); ++i) {
    result_objs.push_back(game_of_life_row(pim_board, i, tmp_pim_obj));
  }

  for(size_t i=0; i<src_host.size(); ++i) {
    print_pim_obj(result_objs[i], width);
  }
  
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Running PIM game of life for board: " << params.width << "x" << params.height << "\n";
  std::vector<std::vector<uint8_t>> x, y;
  if (params.inputFile == nullptr)
  {
    x = {{1,0,1,0,0},
         {0,1,1,1,0},
         {0,0,0,1,1},
         {1,1,0,1,0},
         {1,0,0,1,1}};
    for(size_t i=0; i<5; ++i) {
    std::cout << unsigned(x[0][i]) << ", ";
  }
  std::cout << std::endl;
    // srand((unsigned)time(NULL));
    // x.resize(params.height);
    // for(size_t i=0; i<params.height; ++i) {
    //   x[i].resize(params.width);
    //   for(size_t j=0; j<params.width; ++j) {
    //     x[i][j] = rand() & 1;
    //   }
    // }
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

  //TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iteration.
  game_of_life(x, y);

  // if (params.shouldVerify) 
  // {
  //   // verify result
  //   #pragma omp parallel for
  //   bool is_correct = true;
  //   for (unsigned i = 0; i < params.vectorLength; ++i)
  //   {
  //     if (Y_device[i] != X[i])
  //     {
  //       std::cout << "Wrong answer: " << Y_device[i] << " (expected " << X[i] << ")" << std::endl;
  //       is_correct = false;
  //     }
  //   }
  //   if(is_correct) {
  //     std::cout << "Correct for copy!" << std::endl;
  //   }
  // }

  pimShowStats();

  return 0;
}