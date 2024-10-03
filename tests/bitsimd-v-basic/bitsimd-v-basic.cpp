// Test: BitSIMD-V basic tests
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <cassert>


bool createPimDevice()
{
  unsigned numSubarrayPerBank = 8;
  unsigned numRows = 128;
  unsigned numCols = 256;

  PimStatus status = pimCreateDevice(PIM_DEVICE_BITSIMD_V, 1, 1, numSubarrayPerBank, numRows, numCols);
  assert(status == PIM_OK);
  return true;
}

bool testMicroOps()
{
  unsigned numElements = 1000;

  std::vector<int> src1(numElements);
  std::vector<int> src2(numElements);
  std::vector<int> src3(numElements);
  std::vector<int> dest(numElements);

  PimObjId obj1 = pimAlloc(PIM_ALLOC_V1, numElements, PIM_INT32);
  assert(obj1 != -1);
  PimObjId obj2 = pimAllocAssociated(obj1, PIM_INT32);
  assert(obj2 != -1);
  PimObjId obj3 = pimAllocAssociated(obj1, PIM_INT32);
  assert(obj3 != -1);
  PimObjId obj4 = pimAllocAssociated(obj1, PIM_INT32);
  assert(obj4 != -1);

  // assign some initial values
  for (unsigned i = 0; i < numElements; ++i) {
    src1[i] = i;
    src2[i] = i * 2 - 10;
    src3[i] = i * 3;
  }

  PimStatus status = pimCopyHostToDevice((void*)src1.data(), obj1);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)src2.data(), obj2);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice((void*)src3.data(), obj3);
  assert(status == PIM_OK);

  // row read/write
  status = pimOpReadRowToSa(obj1, 0);
  assert(status == PIM_OK);
  status = pimOpWriteSaToRow(obj4, 0);
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(obj4, (void*)dest.data());
  assert(status == PIM_OK);
  for (unsigned i = 0; i < numElements; ++i) {
    if ((dest[i] & 1) != (src1[i] & 1)) {
      std::cout << "row read/write failed" << std::endl;
      return false;
    }
  }

  // set
  status = pimOpSet(obj1, PIM_RREG_SA, 0);
  assert(status == PIM_OK);
  status = pimOpWriteSaToRow(obj4, 0);
  assert(status == PIM_OK);
  status = pimOpSet(obj1, PIM_RREG_SA, 1);
  assert(status == PIM_OK);
  status = pimOpWriteSaToRow(obj4, 1);
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(obj4, (void*)dest.data());
  assert(status == PIM_OK);
  for (unsigned i = 0; i < numElements; ++i) {
    if ((dest[i] & 1) != 0 || (dest[i] & 2) == 0) {
      std::cout << "set failed" << std::endl;
      return false;
    }
  }

  // move
  status = pimOpReadRowToSa(obj1, 0);
  assert(status == PIM_OK);
  status = pimOpMove(obj1, PIM_RREG_SA, PIM_RREG_R2);
  assert(status == PIM_OK);
  status = pimOpSet(obj1, PIM_RREG_SA, 0);
  assert(status == PIM_OK);
  status = pimOpMove(obj1, PIM_RREG_R2, PIM_RREG_SA);
  assert(status == PIM_OK);
  status = pimOpWriteSaToRow(obj4, 0);
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(obj4, (void*)dest.data());
  assert(status == PIM_OK);
  for (unsigned i = 0; i < numElements; ++i) {
    if ((dest[i] & 1) != (src1[i] & 1)) {
      std::cout << "move failed" << std::endl;
      return false;
    }
  }

  // move
  status = pimOpReadRowToSa(obj1, 0);
  assert(status == PIM_OK);
  status = pimOpMove(obj1, PIM_RREG_SA, PIM_RREG_R1);
  assert(status == PIM_OK);
  status = pimOpReadRowToSa(obj2, 1);
  assert(status == PIM_OK);
  status = pimOpMove(obj1, PIM_RREG_SA, PIM_RREG_R2);
  assert(status == PIM_OK);
  status = pimOpReadRowToSa(obj3, 2);
  assert(status == PIM_OK);
  status = pimOpMove(obj1, PIM_RREG_SA, PIM_RREG_R3);
  assert(status == PIM_OK);
  status = pimOpMove(obj1, PIM_RREG_R2, PIM_RREG_SA);
  assert(status == PIM_OK);
  status = pimOpWriteSaToRow(obj4, 1);
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(obj4, (void*)dest.data());
  assert(status == PIM_OK);
  for (unsigned i = 0; i < numElements; ++i) {
    if (static_cast<bool>(dest[i] & 2) != static_cast<bool>(src2[i] & 2)) {
      std::cout << "move 2 failed" << std::endl;
      return false;
    }
  }

  // logic
  status = pimOpNot(obj1, PIM_RREG_R1, PIM_RREG_SA);
  assert(status == PIM_OK);
  status = pimOpWriteSaToRow(obj4, 0);
  assert(status == PIM_OK);
  status = pimOpAnd(obj1, PIM_RREG_R1, PIM_RREG_R2, PIM_RREG_SA);
  assert(status == PIM_OK);
  status = pimOpWriteSaToRow(obj4, 1);
  assert(status == PIM_OK);
  status = pimOpOr(obj1, PIM_RREG_R1, PIM_RREG_R2, PIM_RREG_SA);
  assert(status == PIM_OK);
  status = pimOpWriteSaToRow(obj4, 2);
  assert(status == PIM_OK);
  status = pimOpNand(obj1, PIM_RREG_R1, PIM_RREG_R2, PIM_RREG_SA);
  assert(status == PIM_OK);
  status = pimOpWriteSaToRow(obj4, 3);
  assert(status == PIM_OK);
  status = pimOpNor(obj1, PIM_RREG_R1, PIM_RREG_R2, PIM_RREG_SA);
  assert(status == PIM_OK);
  status = pimOpWriteSaToRow(obj4, 4);
  assert(status == PIM_OK);
  status = pimOpXor(obj1, PIM_RREG_R1, PIM_RREG_R2, PIM_RREG_SA);
  assert(status == PIM_OK);
  status = pimOpWriteSaToRow(obj4, 5);
  assert(status == PIM_OK);
  status = pimOpXnor(obj1, PIM_RREG_R1, PIM_RREG_R2, PIM_RREG_SA);
  assert(status == PIM_OK);
  status = pimOpWriteSaToRow(obj4, 6);
  assert(status == PIM_OK);
  status = pimOpMaj(obj1, PIM_RREG_R1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_SA);
  assert(status == PIM_OK);
  status = pimOpWriteSaToRow(obj4, 7);
  assert(status == PIM_OK);
  status = pimOpSel(obj1, PIM_RREG_R1, PIM_RREG_R2, PIM_RREG_R3, PIM_RREG_SA);
  assert(status == PIM_OK);
  status = pimOpWriteSaToRow(obj4, 8);
  assert(status == PIM_OK);
  status = pimCopyDeviceToHost(obj4, (void*)dest.data());
  assert(status == PIM_OK);
  for (unsigned i = 0; i < numElements; ++i) {
    bool val1 = static_cast<bool>(src1[i] & 1);
    bool val2 = static_cast<bool>(src2[i] & 2);
    bool val3 = static_cast<bool>(src3[i] & 4);
    if (static_cast<bool>(dest[i] & 1) != !val1) {
      std::cout << "not failed" << std::endl;
      return false;
    }
    if (static_cast<bool>(dest[i] & 2) != (val1 & val2)) {
      std::cout << "and failed" << std::endl;
      return false;
    }
    if (static_cast<bool>(dest[i] & 4) != (val1 | val2)) {
      std::cout << "or failed" << std::endl;
      return false;
    }
    if (static_cast<bool>(dest[i] & 8) != !(val1 & val2)) {
      std::cout << "nand failed" << std::endl;
      return false;
    }
    if (static_cast<bool>(dest[i] & 0x10) != !(val1 | val2)) {
      std::cout << "nor failed" << std::endl;
      return false;
    }
    if (static_cast<bool>(dest[i] & 0x20) != (val1 ^ val2)) {
      std::cout << "xor failed" << std::endl;
      return false;
    }
    if (static_cast<bool>(dest[i] & 0x40) != !(val1 ^ val2)) {
      std::cout << "xnor failed" << std::endl;
      return false;
    }
    if (static_cast<bool>(dest[i] & 0x80) != ((val1 & val2) || (val1 & val3) || (val2 & val3))) {
      std::cout << "maj failed" << std::endl;
      return false;
    }
    if (static_cast<bool>(dest[i] & 0x100) != (val1 ? val2 : val3)) {
      std::cout << "sel failed" << std::endl;
      return false;
    }
  }

  // rotateR
  for (int i = 0; i < 32; ++i) {
    status = pimOpReadRowToSa(obj1, i);
    assert(status == PIM_OK);
    status = pimOpMove(obj1, PIM_RREG_SA, PIM_RREG_R1);
    assert(status == PIM_OK);
    status = pimOpRotateRH(obj1, PIM_RREG_R1);
    assert(status == PIM_OK);
    status = pimOpMove(obj1, PIM_RREG_R1, PIM_RREG_SA);
    assert(status == PIM_OK);
    status = pimOpWriteSaToRow(obj4, i);
    assert(status == PIM_OK);
  }
  status = pimCopyDeviceToHost(obj4, (void*)dest.data());
  assert(status == PIM_OK);
  for (unsigned i = 1; i < numElements; ++i) {
    if (dest[i] != src1[i - 1]) {
      std::cout << "rotateR failed" << std::endl;
      return false;
    }
  }
  if (dest.front() != src1.back()) {
    std::cout << "rotateR failed 2" << std::endl;
    return false;
  }

  // rotateL
  for (int i = 0; i < 32; ++i) {
    status = pimOpReadRowToSa(obj1, i);
    assert(status == PIM_OK);
    status = pimOpMove(obj1, PIM_RREG_SA, PIM_RREG_R1);
    assert(status == PIM_OK);
    status = pimOpRotateLH(obj1, PIM_RREG_R1);
    assert(status == PIM_OK);
    status = pimOpMove(obj1, PIM_RREG_R1, PIM_RREG_SA);
    assert(status == PIM_OK);
    status = pimOpWriteSaToRow(obj4, i);
    assert(status == PIM_OK);
  }
  status = pimCopyDeviceToHost(obj4, (void*)dest.data());
  assert(status == PIM_OK);
  for (unsigned i = 1; i < numElements; ++i) {
    if (dest[i - 1] != src1[i]) {
      std::cout << "rotateL failed " << std::endl;
      return false;
    }
  }
  if (dest.back() != src1.front()) {
    std::cout << "rotateL failed 2" << std::endl;
    return false;
  }

  return true;
}

int main()
{
  std::cout << "PIM test: BitSIMD-V basic" << std::endl;

  bool ok = createPimDevice();
  if (!ok) {
    std::cout << "PIM device creation failed!" << std::endl;
    return 1;
  }

  ok = testMicroOps();
  if (!ok) {
    std::cout << "Test failed!" << std::endl;
    return 1;
  }

  pimShowStats();
  std::cout << "All correct!" << std::endl;
  return 0;
}

