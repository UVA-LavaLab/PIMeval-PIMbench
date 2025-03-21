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
#include <list>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "util.h"
#include "utilBaselines.h"
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

void game_of_life_row(const std::list<PimObjId> &pim_board, const std::list<PimObjId> &pim_board_sums, PimObjId tmp_pim_obj, PimObjId tmp_pim_obj_2, PimObjId tmp_pim_bool, PimObjId result_obj, const bool is_start, const bool is_end) {
  PimStatus status;
  auto it = pim_board.cbegin();
  if(!is_start) {
    it++;
    it++;
    it++;
  }
  // Could be replaced with the ability to add PIM_BOOL
  status = pimBitSliceInsert(*it, tmp_pim_obj_2, 0);
  assert (status == PIM_OK);

  status = pimCopyObjectToObject(tmp_pim_obj_2, tmp_pim_obj);
  assert (status == PIM_OK);
  it++;
  it++;
  status = pimBitSliceInsert(*it, tmp_pim_obj_2, 0);
  assert (status == PIM_OK);

  status = pimAdd(tmp_pim_obj, tmp_pim_obj_2, tmp_pim_obj);
  assert (status == PIM_OK);

  if(!is_start) {
    status = pimAdd(tmp_pim_obj, pim_board_sums.front(), tmp_pim_obj);
    assert (status == PIM_OK);
  }
  if(!is_end) {
    status = pimAdd(tmp_pim_obj, pim_board_sums.back(), tmp_pim_obj);
    assert (status == PIM_OK);
  }

  status = pimEQScalar(tmp_pim_obj, result_obj, 3);
  assert (status == PIM_OK);

  status = pimEQScalar(tmp_pim_obj, tmp_pim_bool, 2);
  assert (status == PIM_OK);
  it--;
  status = pimAnd(tmp_pim_bool, *it, tmp_pim_bool);
  assert (status == PIM_OK);

  status = pimOr(result_obj, tmp_pim_bool, result_obj);
  assert (status == PIM_OK);
}

void add_vector_to_grid(const std::vector<uint8_t> &to_add, PimObjId to_associate, PimObjId tmp_pim_obj_2, std::list<PimObjId> &pim_board, std::list<PimObjId>& pim_board_sums) {

  PimObjId mid = pimAllocAssociated(to_associate, PIM_BOOL);
  assert(mid != -1);
  PimStatus status = pimCopyHostToDevice((void *)to_add.data(), mid);
  assert (status == PIM_OK);

  PimObjId left = pimAllocAssociated(mid, PIM_BOOL);
  assert(left != -1);
  status = pimCopyDeviceToDevice(mid, left); // Could simulator support bool to bool copy object to object?
  assert (status == PIM_OK);
  status = pimShiftElementsRight(left);
  assert (status == PIM_OK);

  

  PimObjId right = pimAllocAssociated(mid, PIM_BOOL);
  assert(right != -1);
  status = pimCopyDeviceToDevice(mid, right); // Could simulator support bool to bool copy object to object?
  assert (status == PIM_OK);
  status = pimShiftElementsLeft(right);
  assert (status == PIM_OK);

  pim_board.push_back(left);
  pim_board.push_back(mid);
  pim_board.push_back(right);

  // Cache sums to reduce repeated work
  // Would be more efficient/cleaner with the ability to add PIM_BOOL+PIM_UINT8=PIM_UINT8
  PimObjId sum = pimAllocAssociated(to_associate, PIM_UINT8);
  assert(mid != -1);
  
  status = pimBitSliceInsert(left, tmp_pim_obj_2, 0);
  assert (status == PIM_OK);

  status = pimCopyObjectToObject(tmp_pim_obj_2, sum);
  assert (status == PIM_OK);

  status = pimBitSliceInsert(mid, tmp_pim_obj_2, 0);
  assert (status == PIM_OK);

  status = pimAdd(sum, tmp_pim_obj_2, sum);
  assert (status == PIM_OK);

  status = pimBitSliceInsert(right, tmp_pim_obj_2, 0);
  assert (status == PIM_OK);

  status = pimAdd(sum, tmp_pim_obj_2, sum);
  assert (status == PIM_OK);

  pim_board_sums.push_back(sum);
}

void game_of_life(const std::vector<std::vector<uint8_t>> &src_host, std::vector<std::vector<uint8_t>> &dst_host)
{
  PimStatus status;
  size_t width = src_host[0].size();
  size_t height = src_host.size();

  PimObjId tmp_pim_obj = pimAlloc(PIM_ALLOC_AUTO, width, PIM_UINT8);
  assert(tmp_pim_obj != -1);

  // tmp_pim_obj_2 is used for adding bools - bools must be converted into UINT8 via splicing before addition
  PimObjId tmp_pim_obj_2 = pimAllocAssociated(tmp_pim_obj, PIM_UINT8);
  assert(tmp_pim_obj_2 != -1);
  status = pimBroadcastUInt(tmp_pim_obj_2, 0);
  assert (status == PIM_OK);

  PimObjId tmp_pim_bool = pimAllocAssociated(tmp_pim_obj, PIM_BOOL);
  assert(tmp_pim_bool != -1);


  std::list<PimObjId> pim_board;
  std::list<PimObjId> pim_board_sums;

  add_vector_to_grid(src_host[0], tmp_pim_obj, tmp_pim_obj_2, pim_board, pim_board_sums);
  if(height > 1) {
    add_vector_to_grid(src_host[1], tmp_pim_obj, tmp_pim_obj_2, pim_board, pim_board_sums);
  }

  PimObjId result_object = pimAllocAssociated(tmp_pim_obj, PIM_BOOL);
  assert(result_object != -1);

  for(size_t i=0; i<height; ++i) {
    game_of_life_row(pim_board, pim_board_sums, tmp_pim_obj, tmp_pim_obj_2, tmp_pim_bool, result_object, i==0, i+1==height);
    status = pimCopyDeviceToHost(result_object, dst_host[i].data());
    assert (status == PIM_OK);
    if(i!=0) {
      for(int i=0; i<3; ++i) {
        pimFree(pim_board.front());
        pim_board.pop_front();
      }
      pimFree(pim_board_sums.front());
      pim_board_sums.pop_front();
    }
    if(i+2<height) {
      add_vector_to_grid(src_host[i+2], tmp_pim_obj, tmp_pim_obj_2, pim_board, pim_board_sums);
    }
  }

  pimFree(tmp_pim_obj);
  pimFree(tmp_pim_obj_2);
  pimFree(tmp_pim_bool);
  while(!pim_board.empty()) {
    pimFree(pim_board.front());
    pim_board.pop_front();
  }
  while(!pim_board_sums.empty()) {
    pimFree(pim_board_sums.front());
    pim_board_sums.pop_front();
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
    getMatrixBool(params.height, params.width, x);
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

  y.resize(params.height, std::vector<uint8_t>(params.width, 0));

  // TODO: Check if vector can fit in one iteration. Otherwise need to run in multiple iterations.
  game_of_life(x, y);

  if (params.shouldVerify) 
  {
    bool is_correct = true;
#pragma omp parallel for
    for(uint64_t i=0; i<y.size(); ++i) {
      for(uint64_t j=0; j<y[0].size(); ++j) {
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
          std::cout << "Wrong answer: " << unsigned(y[i][j]) << " (expected " << unsigned(res_cpu) << ") at position " << i << ", " << j << std::endl;
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