// Test: C++ version of vector multiplication
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <getopt.h>
#include <iostream>
#include <stdint.h>
#include <vector>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../../../../libpimeval/src/libpimeval.h"
#include "../spmv-util.h"

using namespace std;

const size_t SIZE_UBOUND = 32;

void pimCSRSpmv(const std::vector<float> &values, size_t valuesLength,
                const std::vector<float> &x, size_t xLength,
                std::vector<u64> rowPointers, size_t rowPointersLength,
                std::vector<u64> colPointers, size_t colPointersLength,
                std::vector<float> *dst) {
  /*u64 blocks_per_row =*/
  /*    static_cast<u64>(std::ceil((1.0 * x.size()) / (BLOCK_LEN)));*/

  std::array<std::vector<float>, SIZE_UBOUND> valrow;
  std::array<std::vector<float>, SIZE_UBOUND> xrow;
  for (size_t i = 0; i < SIZE_UBOUND; i++) {
    valrow[i].resize(valuesLength);
    xrow[i].resize(valuesLength);
  }

  std::vector<u64> seg_pointers(1);
  u64 segment_count = 0;
  u64 deepest_segment = 0;

  std::cout << "Row pointers length:" << rowPointersLength << std::endl;
  for (size_t row = 0; row < rowPointersLength - 1; row++) {
    // calculate the number of segments in the row
    u64 segments =
        static_cast<u64>(ceil(1.0f * (rowPointers[row + 1] - rowPointers[row]) /
                              (1.0f * SIZE_UBOUND)));

    for (u64 segment = 0; segment < segments; segment++) {
      for (size_t row_pointer = rowPointers[row] + segment * SIZE_UBOUND;
           row_pointer < rowPointers[row + 1] &&
           row_pointer < rowPointers[row] + (segment + 1) * SIZE_UBOUND;
           row_pointer++) {
        if (segment_count + segment > values.size()) {
          std::cout << "Failed at row ptr: " << row_pointer
                    << ", segment count: " << segment_count
                    << ", and segment: " << segment << std::endl;
          std::cout << "Segments: " << segments << std::endl;
          std::cout << "Row: " << row << std::endl;
          std::cout << "Values size: " << values.size() << std::endl;
          std::cout << "Row pointers end: "
                    << rowPointers[rowPointers.size() - 1] << std::endl;
          exit(1);
        }
        /*std::cout << "Write location [i1]: "*/
        /*          << row_pointer - rowPointers[row] - segment * SIZE_UBOUND*/
        /*          << ", [i2]: " << segment_count + segment << std::endl;*/
        valrow[row_pointer - rowPointers[row] - segment * SIZE_UBOUND]
              [segment_count + segment] = values[row_pointer];
        fprintf(stderr, "colPointers[row_pointer] %lu",
                colPointers[row_pointer]);
        xrow[row_pointer - rowPointers[row] - segment * SIZE_UBOUND]
            [segment_count + segment] = x[colPointers[row_pointer]];
        if (row_pointer - rowPointers[row] > deepest_segment)
          deepest_segment =
              std::min(row_pointer - rowPointers[row], SIZE_UBOUND - 1);
      }

      segment_count += segments;
    }
    seg_pointers.push_back(segment_count);
  }
  std::vector<float> xRearranged(valuesLength);

  for (u64 i = 0; i < colPointersLength; i++) {
    /*status = pimCopyHostToDevice((void *)&x.data()[colPointers[i]], xDevice,
     * i,*/
    /*                             i + 1);*/
    /*assert(status == PIM_OK);*/
    xRearranged[i] = x[colPointers[i]];
  }

  /*for (size_t i = 0; i < valuesLength; i++)*/
  /*  std::cout << values[i] << " ";*/
  /*std::cout << std::endl;*/
  /*for (size_t i = 0; i < valuesLength; i++)*/
  /*  std::cout << xRearranged[i] << " ";*/
  /*std::cout << std::endl;*/
  /*std::cout << "----------------" << std::endl;*/
  /*std::cout << deepest_segment << std::endl;*/
  /*for (size_t i = 0; i <= deepest_segment; i++) {*/
  /*  std::cout << "index: " << i << std::endl;*/
  /*  for (size_t j = 0; j < segment_count; j++)*/
  /*    std::cout << valrow[i][j] << " ";*/
  /*  std::cout << std::endl;*/
  /*  for (size_t j = 0; j < segment_count; j++)*/
  /*    std::cout << xrow[i][j] << " ";*/
  /*  std::cout << std::endl;*/
  /*  std::cout << "-" << std::endl;*/
  /*}*/

  for (size_t i = 0; i < SIZE_UBOUND; i++) {
    valrow[i].resize(segment_count);
    xrow[i].resize(segment_count);
  }

  PimStatus status;

  std::vector<PimObjId> valuesDevice(deepest_segment + 1);
  std::vector<PimObjId> xDevice(deepest_segment + 1);

  valuesDevice[0] = pimAlloc(PIM_ALLOC_AUTO, segment_count, PIM_INT32);

  xDevice[0] = pimAllocAssociated(valuesDevice[0], PIM_INT32);

  status = pimCopyHostToDevice((void *)valrow[0].data(), valuesDevice[0]);
  assert(status == PIM_OK);

  status = pimCopyHostToDevice((void *)xrow[0].data(), xDevice[0]);
  assert(status == PIM_OK);

  for (size_t i = 1; i <= deepest_segment; i++) {
    valuesDevice[i] = pimAllocAssociated(valuesDevice[0], PIM_INT32);
    if (valuesDevice[i] == -1) {
      std::cout << "PIM allocation failed at index " << i << std::endl;
      return;
    }

    status = pimCopyHostToDevice((void *)valrow[i].data(), valuesDevice[i]);
    assert(status == PIM_OK);

    xDevice[i] = pimAllocAssociated(valuesDevice[i], PIM_INT32);
    if (xDevice[i] == -1) {
      std::cout << "PIM allocation failed at index " << i << std::endl;
      return;
    }

    status = pimCopyHostToDevice((void *)xrow[i].data(), xDevice[i]);
    assert(status == PIM_OK);
  }

  for (size_t i = 0; i <= deepest_segment; i++) {
    status = pimMul(xDevice[i], valuesDevice[i], valuesDevice[i]);
    assert(status == PIM_OK);
  }

  for (size_t i = 1; i <= deepest_segment; i++) {
    status = pimAdd(valuesDevice[i], valuesDevice[0], valuesDevice[0]);
    assert(status == PIM_OK);
  }

  /*std::cout << "----------------" << std::endl;*/
  /*dst->resize(rowPointersLength - 1);*/
  /*std::cout << dst->size() << " vs " << seg_pointers.size() << std::endl;*/
  /*std::cout << "----------------" << std::endl;*/
  /*for (auto &s : seg_pointers)*/
  /*  std::cout << s << " ";*/
  /*std::cout << std::endl;*/

  /*std::vector<i32> seg_ptr_img(seg_pointers.size());*/
  float temp = 0;
  for (size_t i = 0; i < seg_pointers.size() - 1; i++) {
    if (seg_pointers[i + 1] - seg_pointers[i] != 0) {
      // the following code does not always evaluate as expected
      /*if (seg_pointers[i + 1] - seg_pointers[i] == 1) {*/
      /*  status =*/
      /*      pimCopyDeviceToHost(valuesDevice[0], (void *)&(*dst).data()[i]),*/
      /*                          seg_pointers[i], seg_pointers[i] + 1);*/
      /*  assert(status == PIM_OK);*/
      /*} else {*/
      status = pimRedSum(valuesDevice[0], (void *)&temp, seg_pointers[i],
                         seg_pointers[i + 1]);
      assert(status == PIM_OK);
      (*dst)[i] = temp;
      /*}*/
    }

    std::cout << "row pointers len " << rowPointersLength << std::endl;
    std::cout << "destination vec " << (*dst)[i] << std::endl;
    std::cout << "index " << i << std::endl;
  }

  for (size_t i = 0; i < deepest_segment; i++) {
    pimFree(valuesDevice[i]);
    pimFree(xDevice[i]);
  }
}

int main(int argc, char *argv[]) {
  struct Params params = getInputParams(argc, argv);
  // std::cout << "Vector length: " << params.vectorLength << "\n";
  std::vector<float> x, values, Y, dst;
  std::vector<u64> matDim, rowPointers, colPointers;
  rowPointers.resize(params.matRowsBlocks * BLOCK_LEN + 1);
  colPointers.resize(params.nnz);
  values.resize(params.nnz);
  x.resize(params.matColsBlocks * BLOCK_LEN);
  // std::vector<int> src1, src2, dst;
  if (params.inputFile == nullptr) {
    u64 vert = params.matRowsBlocks * BLOCK_LEN,
        hori = params.matColsBlocks * BLOCK_LEN;
    generateUniformSpMV(&rowPointers, &colPointers, &values, &x, vert, hori,
                        params.nnz, (1.0f * params.nnz) / (1.0f * vert * hori));
    /*generateSpMV(params, &x, &values, &rowPointers, &colPointers);*/
    if (params.shouldVerify) {
      cpuCSRSpmv(values, values.size(), x, x.size(), rowPointers,
                 rowPointers.size(), colPointers, colPointers.size(), Y);
    }
  } else {
    std::cout << "Input file support not implemented" << std::endl;
    return 1;
  }
  if (!pimCreateDeviceFromConfig(params.deviceType, params.configFile))
    return 1;

  // get device properties
  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  if (status != PIM_OK) {
    std::cout << "Abort: pimGetDeviceProperties failed" << std::endl;
    return 1;
  }

  std::cout << "Starting PIM calculation" << std::endl;
  dst.resize(rowPointers.size() - 1);
  pimCSRSpmv(values, values.size(), x, x.size(), rowPointers,
             rowPointers.size(), colPointers, colPointers.size(), &dst);

  if (params.shouldVerify) {
    std::cout << "Verifying results of computation" << std::endl;
    bool ret_false = false;
    for (u64 i = 0; i < dst.size(); i++) {
      if (dst[i] != Y[i]) {
        ret_false = true;
        std::cout << "Index " << i
                  << " of returned vector does not match test case's output."
                  << std::endl;
        std::cout << "Returned output: " << dst[i] << std::endl;
        std::cout << "Intended output: " << Y[i] << std::endl;
      }
    }
    if (ret_false) {
      /*std::cout << std::endl;*/
      /*for (const auto &i : values)*/
      /*  std::cout << i << " ";*/
      /*std::cout << std::endl;*/
      return 1;
    }
  }

  pimShowStats();

  fprintf(stderr, "Dst size: %ld", dst.size());

  return 0;
}
