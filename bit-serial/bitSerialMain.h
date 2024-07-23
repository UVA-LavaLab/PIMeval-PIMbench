// Bit-Serial Performance Modeling - Main
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef BIT_SERIAL_MAIN_H
#define BIT_SERIAL_MAIN_H

#include <string>
#include <vector>

//! @class  bitSerialMain
//! @brief  Bit-serial performance tests main entry
class bitSerialMain
{
public:
  bitSerialMain();
  ~bitSerialMain() {}

  void runTests(const std::vector<std::string>& deviceList = {}, const std::vector<std::string>& testList = {});

  const std::vector<std::string>& getDeviceList() const { return m_deviceList; }
  const std::vector<std::string>& getTestList() const { return m_testList; }

private:
  std::vector<std::string> m_deviceList;
  std::vector<std::string> m_testList;
};

#endif

