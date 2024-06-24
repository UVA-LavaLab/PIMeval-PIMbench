// Bit-Serial Performance Modeling - Main
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "bitSerialMain.h"
#include "bitSerialBase.h"
#include "bitSerialBitsimd.h"
#include "bitSerialBitsimdAp.h"
#include "bitSerialSimdram.h"
#include <iostream>

bool
bitSerialMain::testBitsimd()
{
  std::cout << "========== PIM_DEVICE_BITSIMD_V ==========" << std::endl;
  bitSerialBase* model = new bitSerialBitsimd;
  bool ok = model->runTests();
  delete model;
  std::cout << (ok ? "Passed" : "Failed") << std::endl;
  return ok;
}

bool
bitSerialMain::testBitsimdAp()
{
  std::cout << "========== PIM_DEVICE_BITSIMD_V_AP ==========" << std::endl;
  bitSerialBase* model = new bitSerialBitsimdAp;
  bool ok = model->runTests();
  delete model;
  std::cout << (ok ? "Passed" : "Failed") << std::endl;
  return ok;
}

bool
bitSerialMain::testSimdram()
{
  std::cout << "========== PIM_DEVICE_SIMDRAM ==========" << std::endl;
  bitSerialBase* model = new bitSerialSimdram;
  bool ok = model->runTests();
  delete model;
  std::cout << (ok ? "Passed" : "Failed") << std::endl;
  return ok;
}

bool
bitSerialMain::runAllTests()
{
  std::cout << "Bit Serial Performance Modeling" << std::endl;
  bool ok = true;
  ok &= testBitsimd();
  return ok;
}

int main()
{
  bitSerialMain app;
  bool ok = app.runAllTests();
  std::cout << (ok ? "All passed!" : "Some failed!") << std::endl;
  return 0;
}

