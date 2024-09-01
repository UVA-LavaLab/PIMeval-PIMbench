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

void game_of_life(const std::vector<std::vector<uint8_t>> &src_host, std::vector<std::vector<uint8_t>> &dst_host)
{
  unsigned bitsPerElement = 8;
  size_t width = src_host[0].size();
  size_t height = src_host.size();
  PimObjId sum_pim = pimAlloc(PIM_ALLOC_AUTO, width, bitsPerElement, PIM_UINT8);
  assert(sum_pim != -1);

  PimObjId res_pim = pimAllocAssociated(bitsPerElement, sum_pim, PIM_UINT8);
  assert(res_pim != -1);
  PimStatus status = pimCopyHostToDevice((void *)src_host[1].data(), res_pim);
  assert (status == PIM_OK);

  PimObjId upper_left = pimAllocAssociated(bitsPerElement, sum_pim, PIM_UINT8);
  assert(upper_left != -1);
  // Should be able to use a ranged ref to replace shift once implemented
  status = pimCopyHostToDevice((void *)src_host[0].data(), upper_left);
  assert (status == PIM_OK);
  status = pimShiftElementsRight(upper_left);
  assert (status == PIM_OK);

  PimObjId upper_mid = pimAllocAssociated(bitsPerElement, sum_pim, PIM_UINT8);
  assert(upper_mid != -1);
  status = pimCopyHostToDevice((void *)src_host[0].data(), upper_mid);
  assert (status == PIM_OK);

  PimObjId upper_right = pimAllocAssociated(bitsPerElement, sum_pim, PIM_UINT8);
  assert(upper_right != -1);
  status = pimCopyHostToDevice((void *)src_host[0].data(), upper_right);
  assert (status == PIM_OK);
  status = pimShiftElementsLeft(upper_right);
  assert (status == PIM_OK);

  PimObjId mid_left = pimAllocAssociated(bitsPerElement, sum_pim, PIM_UINT8);
  assert(mid_left != -1);
  status = pimCopyHostToDevice((void *)src_host[1].data(), mid_left);
  assert (status == PIM_OK);
  status = pimShiftElementsRight(mid_left);
  assert (status == PIM_OK);

  PimObjId mid_right = pimAllocAssociated(bitsPerElement, sum_pim, PIM_UINT8);
  assert(mid_right != -1);
  status = pimCopyHostToDevice((void *)src_host[1].data(), mid_right);
  assert (status == PIM_OK);
  status = pimShiftElementsLeft(mid_right);
  assert (status == PIM_OK);

  PimObjId lower_left = pimAllocAssociated(bitsPerElement, sum_pim, PIM_UINT8);
  assert(lower_left != -1);
  status = pimCopyHostToDevice((void *)src_host[2].data(), lower_left);
  assert (status == PIM_OK);
  status = pimShiftElementsRight(lower_left);
  assert (status == PIM_OK);

  PimObjId lower_mid = pimAllocAssociated(bitsPerElement, sum_pim, PIM_UINT8);
  assert(lower_mid != -1);
  status = pimCopyHostToDevice((void *)src_host[2].data(), lower_mid);
  assert (status == PIM_OK);

  PimObjId lower_right = pimAllocAssociated(bitsPerElement, sum_pim, PIM_UINT8);
  assert(lower_right != -1);
  status = pimCopyHostToDevice((void *)src_host[2].data(), lower_right);
  assert (status == PIM_OK);
  status = pimShiftElementsLeft(lower_right);
  assert (status == PIM_OK);

  pimAdd(upper_left, upper_mid, sum_pim);
  pimAdd(upper_right, sum_pim, sum_pim);
  pimAdd(mid_left, sum_pim, sum_pim);
  pimAdd(mid_right, sum_pim, sum_pim);
  pimAdd(lower_left, sum_pim, sum_pim);
  pimAdd(lower_mid, sum_pim, sum_pim);
  pimAdd(lower_right, sum_pim, sum_pim);

  print_pim_obj(sum_pim, width);
  print_pim_obj(res_pim, width);

  PimObjId tmp_pim_obj = pimAllocAssociated(bitsPerElement, sum_pim, PIM_UINT8);
  assert(tmp_pim_obj != -1);

  status = pimEQScalar(sum_pim, tmp_pim_obj, 2);
  assert (status == PIM_OK);

  status = pimAnd(tmp_pim_obj, res_pim, res_pim);
  assert (status == PIM_OK);

  // Lives if 3 neighbors, or (2 and alive)
  status = pimEQScalar(sum_pim, tmp_pim_obj, 3);
  assert (status == PIM_OK);

  status = pimOr(tmp_pim_obj, res_pim, res_pim);
  assert (status == PIM_OK);

  

  std::cout << "final state: ";
  print_pim_obj(res_pim, width);
  
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
         {0,0,0,1,1}};
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