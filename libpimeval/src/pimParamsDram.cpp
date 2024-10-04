// File: pimParamsDram.cc
// PIMeval Simulator - DRAM parameters
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimParamsDram.h"
#include "pimUtils.h"
#include "pimParamsDDRDram.h"
#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <locale>
#include <stdexcept>

// Static factory method to create appropriate subclass based on protocol enum
std::unique_ptr<pimParamsDram> pimParamsDram::create(PimDeviceProtocolEnum deviceProtocol)
{
  if (deviceProtocol == PIM_DEVICE_PROTOCOL_DDR || deviceProtocol == PIM_DEVICE_PROTOCOL_LPDDR)
  {
    return std::make_unique<pimParamsDDRDram>();
  }
  else
  {
    throw std::invalid_argument("Unknown protocol");
  }
}

// Static factory method to create appropriate subclass based on config string
std::unique_ptr<pimParamsDram> pimParamsDram::createFromConfig(const std::string &config)
{
  std::istringstream configStream(config);
  std::string line;
  std::unordered_map<std::string, std::string> params;

  while (std::getline(configStream, line))
  {
    line = pimUtils::removeAfterSemicolon(line);
    if (line.empty() || line[0] == '[')
    {
      continue; // Skip section headers and empty lines
    }

    // Parse key-value pairs
    size_t equalPos = line.find('=');
    if (equalPos != std::string::npos)
    {
      std::string key = line.substr(0, equalPos);
      std::string value = line.substr(equalPos + 1);
      params[pimUtils::trim(key)] = pimUtils::trim(value); // Store in params map
    }
  }

  // Check if the "protocol" key exists
  if (params.find("protocol") == params.end())
  {
    throw std::invalid_argument("Missing protocol key in config file.");
  }

  // Extract protocol from params
  std::string deviceProtocol = params["protocol"];

  // Instantiate the appropriate subclass based on the protocol
  if (deviceProtocol == "DDR3" || deviceProtocol == "DDR4" || deviceProtocol == "DDR5" || deviceProtocol == "LPDDR4")
  {
    return std::make_unique<pimParamsDDRDram>(params);
  }
  else
  {
    throw std::invalid_argument("Unknown protocol: " + deviceProtocol);
  }
}