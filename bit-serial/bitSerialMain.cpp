// Bit-Serial Performance Modeling - Main
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "bitSerialMain.h"
#include "bitSerialBase.h"
#include "bitSerialBitsimd.h"
#include "bitSerialBitsimdAp.h"
#include "bitSerialSimdram.h"
#include <iostream>

bitSerialMain::bitSerialMain()
{
  m_deviceList = {
    "bitsimd_v",
    "bitsimd_v_ap",
    "simdram",
  };
  m_testList = {
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "fp32",
  };
}

void
bitSerialMain::runTests(const std::vector<std::string>& deviceList, const std::vector<std::string>& testList)
{
  const auto& myDeviceList = deviceList.empty() ? m_deviceList : deviceList;
  const auto& myTestList = testList.empty() ? m_testList : testList;

  // device -> test -> numPassed/numTests
  std::map<std::string, std::map<std::string, std::pair<int, int>>> stats;

  for (const auto& device : myDeviceList) {
    std::cout << "INFO: Bit Serial Performance Modeling for " << device << std::endl;
    std::unique_ptr<bitSerialBase> model;
    if (device == "bitsimd_v") {
      model = std::make_unique<bitSerialBitsimd>();
    } else if (device == "bitsimd_v_ap") {
      model = std::make_unique<bitSerialBitsimdAp>();
    } else if (device == "simdram") {
      model = std::make_unique<bitSerialSimdram>();
    }
    bool ok = false;
    if (model) {
      ok = model->runTests(myTestList);
      stats[device] = model->getStats();
    }
    std::cout << "INFO: Bit Serial Performance Modeling for " << device << (ok ? " -- Succeed" : " -- Failed!") << std::endl;
  }

  // show stats
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Summary (passed / total):" << std::endl;
  for (const auto& device : myDeviceList) {
    std::cout << "    " << device << std::endl;
    auto it = stats.find(device);
    if (it != stats.end()) {
      for (const auto& test : myTestList) {
        auto it2 = it->second.find(test);
        if (it2 != it->second.end()) {
          std::cout << "        " << test << " : " << it2->second.first << " / " << it2->second.second << std::endl;
        }
      }
    }
  }
  std::cout << "----------------------------------------" << std::endl;
}

int main()
{
  bitSerialMain app;
  app.runTests();
  return 0;
}

