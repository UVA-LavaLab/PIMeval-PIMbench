// Bit-Serial Performance Modeling - Base
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "bitSerialBase.h"

//! @brief  Create device
void
bitSerialBase::createDevice()
{
  // create with any v-layout PIM device
  PimDeviceEnum deviceType = getDeviceType();
  PimStatus status = pimCreateDevice(deviceType, 1, 1, 16, 8192, 1024);
  assert(status == PIM_OK);
}

//! @brief  Delete device
void
bitSerialBase::deleteDevice()
{
  PimStatus status = pimDeleteDevice();
  assert(status == PIM_OK);
}

//! @brief  Run tests
bool
bitSerialBase::runTests(const std::vector<std::string>& testList)
{
  bool ok = true;

  createDevice();

  for (const auto& testName : testList) {
    if (testName == "int8") {
      ok &= testInt<int8_t>(testName, PIM_INT8);
    } else if (testName == "int16") {
      ok &= testInt<int16_t>(testName, PIM_INT16);
    } else if (testName == "int32") {
      ok &= testInt<int32_t>(testName, PIM_INT32);
    } else if (testName == "int64") {
      ok &= testInt<int64_t>(testName, PIM_INT64);
    } else if (testName == "uint8") {
      ok &= testInt<uint8_t>(testName, PIM_UINT8);
    } else if (testName == "uint16") {
      ok &= testInt<uint16_t>(testName, PIM_UINT16);
    } else if (testName == "uint32") {
      ok &= testInt<uint32_t>(testName, PIM_UINT32);
    } else if (testName == "uint64") {
      ok &= testInt<uint64_t>(testName, PIM_UINT64);
    } else if (testName == "fp32") {
      ok &= testFp<float>(testName, PIM_FP32);
    } else {
      std::cout << "Error: Unknown test " << testName << std::endl;
    }
  }

  deleteDevice();

  return ok;
}

