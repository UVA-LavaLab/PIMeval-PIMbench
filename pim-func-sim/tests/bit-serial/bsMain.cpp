// Bit-Serial Performance Modeling - Main
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "bsMain.h"
#include "bsBase.h"
#include "bsBitsimd.h"
#include "bsBitsimdAp.h"
#include "bsSimdram.h"
#include <iostream>

bool
bsMain::testBitsimd()
{
  std::cout << "========== PIM_DEVICE_BITSIMD_V ==========" << std::endl;
  bsBase* bs = new bsBitsimd;
  bool ok = bs->runTests();
  delete bs;
  std::cout << (ok ? "Passed" : "Failed") << std::endl;
  return ok;
}

bool
bsMain::testBitsimdAp()
{
  std::cout << "========== PIM_DEVICE_BITSIMD_V_AP ==========" << std::endl;
  bsBase* bs = new bsBitsimdAp;
  bool ok = bs->runTests();
  delete bs;
  std::cout << (ok ? "Passed" : "Failed") << std::endl;
  return ok;
}

bool
bsMain::testSimdram()
{
  std::cout << "========== PIM_DEVICE_SIMDRAM ==========" << std::endl;
  bsBase* bs = new bsSimdram;
  bool ok = bs->runTests();
  delete bs;
  std::cout << (ok ? "Passed" : "Failed") << std::endl;
  return ok;
}

bool
bsMain::runAllTests()
{
  std::cout << "Bit Serial Performance Modeling" << std::endl;
  bool ok = true;
  ok &= testBitsimd();
  //ok &= testBitsimdAp();
  //ok &= testSimdram();
  return ok;
}

int main()
{
  bsMain bs;
  bool ok = bs.runAllTests();
  std::cout << (ok ? "All passed!" : "Some failed!") << std::endl;
  return 0;
}

