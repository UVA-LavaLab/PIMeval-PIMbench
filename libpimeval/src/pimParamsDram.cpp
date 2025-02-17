// File: pimParamsDram.cc
// PIMeval Simulator - DRAM parameters
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimParamsDram.h"
#include "pimUtils.h"
#include "pimParamsDDRDram.h"
#include "pimParamsLPDDRDram.h"
#include "pimParamsHBMDram.h"
#include <string>                      // for string
#include <memory>                      // for make_unique, unique_ptr
#include <unordered_map>               // for unordered_map
#include <stdexcept>                   // for invalid_argument


// Static factory method to create appropriate subclass based on protocol enum
std::unique_ptr<pimParamsDram> pimParamsDram::create(PimDeviceProtocolEnum deviceProtocol)
{
  if (deviceProtocol == PIM_DEVICE_PROTOCOL_DDR)
  {
    return std::make_unique<pimParamsDDRDram>();
  }
  else if (deviceProtocol == PIM_DEVICE_PROTOCOL_LPDDR)
  {
    return std::make_unique<pimParamsLPDDRDram>();
  }
  else if (deviceProtocol == PIM_DEVICE_PROTOCOL_HBM)
  {
    return std::make_unique<pimParamsHBMDram>();
  }
  else
  {
    std::string errorMessage("PIM-Error: Inavalid DRAM protocol parameter.\n");
    throw std::invalid_argument(errorMessage);
  }
}

// Static factory method to create appropriate subclass based on config file
std::unique_ptr<pimParamsDram> pimParamsDram::createFromConfig(const std::string& memConfigFilePath)
{
  std::unordered_map<std::string, std::string> params = pimUtils::readParamsFromConfigFile(memConfigFilePath);

  // Check if the "protocol" key exists
  if (params.find("protocol") == params.end())
  {
    std::string errorMessage("PIM-Error: Missing DRAM protocol parameter.\n");
    throw std::invalid_argument(errorMessage);
  }

  // Extract protocol from params
  std::string deviceProtocol = params["protocol"];

  // Instantiate the appropriate subclass based on the protocol
  if (deviceProtocol == "DDR3" || deviceProtocol == "DDR4" || deviceProtocol == "DDR5")
  {
    return std::make_unique<pimParamsDDRDram>(params);
  } 
  else if (deviceProtocol == "LPDDR3" || deviceProtocol == "LPDDR4") {
    return std::make_unique<pimParamsLPDDRDram>(params);
  } 
  else if (deviceProtocol == "HBM" || deviceProtocol == "HBM2") {
    return std::make_unique<pimParamsHBMDram>(params);
  }
  else
  {
    throw std::invalid_argument("Unknown protocol: " + deviceProtocol);
  }
}

