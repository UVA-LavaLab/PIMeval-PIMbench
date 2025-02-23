// File: pimSimConfig.cpp
// PIMeval Simulator - PIM Simulator Configurations
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimSimConfig.h"
#include "pimUtils.h"
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <filesystem>


//! @brief  Init PIMeval simulation configuration parameters at device creation
bool
pimSimConfig::init(PimDeviceEnum deviceType,
    unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank,
    unsigned numRowPerSubarray, unsigned numColPerSubarray)
{
  return deriveConfig(deviceType, "",
                      numRanks, numBankPerRank, numSubarrayPerBank,
                      numRowPerSubarray, numColPerSubarray);
}

//! @brief  Init PIMeval simulation configuration parameters at device creation
bool
pimSimConfig::init(PimDeviceEnum deviceType, const std::string& configFilePath)
{
  return deriveConfig(deviceType, configFilePath);
}

//! @brief  Reset pimSimConfig to uninitialized status
void
pimSimConfig::uninit()
{
  m_simConfigFile.clear();
  m_memConfigFile.clear();
  m_deviceType = PIM_DEVICE_NONE;
  m_simTarget = PIM_DEVICE_NONE;
  m_memoryProtocol = PIM_DEVICE_PROTOCOL_DDR;
  m_numRanks = 0;
  m_numBankPerRank = 0;
  m_numSubarrayPerBank = 0;
  m_numRowPerSubarray = 0;
  m_numColPerSubarray = 0;
  m_numThreads = 0;
  m_analysisMode = false;
  m_debug = 0;
  m_loadBalanced = false;
}

//! @brief  Show all configuration parameters
void
pimSimConfig::show() const
{
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "PIM-Config: Debug Flags = 0x" << std::hex << m_debug << std::dec << std::endl;
  std::cout << "PIM-Config: Simulator Config File: "
            << (m_simConfigFile.empty() ? "<NONE>" : m_simConfigFile) << std::endl;
  std::cout << "PIM-Config: Memory Config File: "
            << (m_memConfigFile.empty() ? "<DEFAULT>" : m_memConfigFile) << std::endl;
  std::cout << "PIM-Config: Memory Protocol: " << pimUtils::pimProtocolEnumToStr(m_memoryProtocol) << std::endl;

  std::cout << "PIM-Config: Current Device = " << pimUtils::pimDeviceEnumToStr(m_deviceType)
            << ", Simulation Target = " << pimUtils::pimDeviceEnumToStr(m_simTarget) << std::endl;

  std::cout << "PIM-Config: #ranks = " << m_numRanks
            << ", #banksPerRank = " << m_numBankPerRank
            << ", #subarraysPerBank = " << m_numSubarrayPerBank
            << ", #rowsPerSubarray = " << m_numRowPerSubarray
            << ", #colsPerSubarray = " << m_numColPerSubarray << std::endl;

  std::cout << "PIM-Config: Number of threads = " << m_numThreads << std::endl;
  std::cout << "PIM-Config: Load Balanced = " << m_loadBalanced << std::endl;
  std::cout << "----------------------------------------" << std::endl;
}

//! @brief  Derive PIMeval simulation configuration parameters with priority rules
bool
pimSimConfig::deriveConfig(PimDeviceEnum deviceType,
    const std::string& configFilePath,
    unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank,
    unsigned numRowPerSubarray, unsigned numColPerSubarray)
{
  bool ok = true;

  // Derive debug flags first
  ok = ok & deriveDebug();

  // Read environment variables
  m_envParams = readEnvVars();

  // Derive simulator config file
  ok = ok & deriveSimConfigFile(configFilePath);

  // Read config file parameters
  m_cfgParams = readSimConfigFileParams();

  // Derive other configuration parameters in order
  ok = ok & deriveDeviceType(deviceType);
  ok = ok & deriveSimTarget();
  ok = ok & deriveMemConfigFile();
  ok = ok & deriveDimensions(numRanks, numBankPerRank, numSubarrayPerBank, numRowPerSubarray, numColPerSubarray);
  ok = ok & deriveNumThreads();
  ok = ok & deriveMiscEnvVars();
  ok = ok & deriveLoadBalance();

  // Show summary
  show();
  if (!ok) {
    std::cout << "PIM-Error: Please resolve incorrect PIMeval configuration." << std::endl;
  }
  return ok;
}

//! @brief  Derive Params: Debug Flags
bool
pimSimConfig::deriveDebug()
{
  m_debug = 0;
  std::string envVal;
  bool hasEnv = pimUtils::getEnvVar(m_envVarDebug, envVal);
  if (hasEnv && !envVal.empty()) {
    bool ok = pimUtils::convertStringToUnsigned(envVal, m_debug);
    if (!ok) {
      std::cout << "PIM-Error: Incorrect environment variable: " << m_envVarDebug << " = " << envVal << std::endl;
      return false;
    }
  }
  return true;
}

//! @brief  Read config env vars
std::unordered_map<std::string, std::string>
pimSimConfig::readEnvVars() const
{
  // Align with env var name list in header
  const std::vector<std::string> envVars = {
    m_envVarSimConfig,
    m_envVarMemConfig,
    m_envVarSimTarget,
    m_envVarNumRanks,
    m_envVarNumBankPerRank,
    m_envVarNumSubarrayPerBank,
    m_envVarNumRowPerSubarray,
    m_envVarNumColPerSubarray,
    m_envVarMaxNumThreads,
    m_envVarAnalysisMode,
    m_envVarDebug,
    m_envVarLoadBalance,
  };

  std::unordered_map<std::string, std::string> params;
  params = pimUtils::readParamsFromEnvVars(envVars);

  if (m_debug & pimSimConfig::DEBUG_PARAMS) {
    for (const auto& [key, val] : params) {
      std::cout << "PIM-Debug: Environment variable: " << key << " = " << val << std::endl;
    }
  }

  return params;
}

//! @brief  Derive Params: Simulator Configuration File
bool
pimSimConfig::deriveSimConfigFile(const std::string& configFilePath)
{
  m_simConfigFile.clear();
  // If a config file is specified through APIs, use it. Otherwise check env var
  if (!configFilePath.empty()) {
    m_simConfigFile = configFilePath;
  } else if (m_envParams.find(m_envVarSimConfig) != m_envParams.end()) {
    m_simConfigFile = m_envParams.at(m_envVarSimConfig);
  }
  if (!m_simConfigFile.empty()) {
    if (!std::filesystem::exists(m_simConfigFile)) {
      std::cout << "PIM-Error: Cannot find simulator config file: " << m_simConfigFile << std::endl;
      return false;
    }
  }
  return true;
}

//! @brief  Read config file params
std::unordered_map<std::string, std::string>
pimSimConfig::readSimConfigFileParams() const
{
  std::unordered_map<std::string, std::string> params;
  if (!m_simConfigFile.empty()) {
    params = pimUtils::readParamsFromConfigFile(m_simConfigFile);

    if (m_debug & pimSimConfig::DEBUG_PARAMS) {
      for (const auto& [key, val] : params) {
        std::cout << "PIM-Debug: Simulator config file parameter: " << key << " = " << val << std::endl;
      }
    }
  }
  return params;
}

//! @brief  Derive Params: Device Type
bool
pimSimConfig::deriveDeviceType(PimDeviceEnum deviceType)
{
  m_deviceType = deviceType;
  return true;
}

//! @brief  Derive Params: Simulation Target
bool
pimSimConfig::deriveSimTarget()
{
  // If device type is not functional, always use it as simulation target
  m_simTarget = m_deviceType;

  if (m_deviceType == PIM_FUNCTIONAL) {
    bool hasVal = false;
    std::string val;
    // Check simulator config file
    if (m_simTarget == PIM_DEVICE_NONE || m_simTarget == PIM_FUNCTIONAL) {
      val = pimUtils::getOptionalParam(m_cfgParams, m_cfgVarSimTarget, hasVal);
      if (hasVal) {
        m_simTarget = pimUtils::strToPimDeviceEnum(val);
        if (m_simTarget == PIM_DEVICE_NONE) {
          std::cout << "PIM-Error: Incorrect config file parameter: " << m_cfgVarSimTarget << "=" << val << std::endl;
          return false;
        }
      }
    }
    // Check env var
    if (m_simTarget == PIM_DEVICE_NONE || m_simTarget == PIM_FUNCTIONAL) {
      val = pimUtils::getOptionalParam(m_envParams, m_envVarSimTarget, hasVal);
      if (hasVal) {
        m_simTarget = pimUtils::strToPimDeviceEnum(val);
        if (m_simTarget == PIM_DEVICE_NONE) {
          std::cout << "PIM-Error: Incorrect environment variable: " << m_envVarSimTarget << "=" << val << std::endl;
          return false;
        }
      }
    }
    // Check macro
    if (m_simTarget == PIM_DEVICE_NONE || m_simTarget == PIM_FUNCTIONAL) {
      // from 'make PIM_SIM_TARGET=...'
      #if defined(PIM_SIM_TARGET)
      m_simTarget = PIM_SIM_TARGET;
      #endif
    }
    // Use default
    if (m_simTarget == PIM_DEVICE_NONE || m_simTarget == PIM_FUNCTIONAL) {
      m_simTarget = DEFAULT_SIM_TARGET;
    }
  }

  return true;
}

//! @brief  Derive Params: Memory Config File
bool
pimSimConfig::deriveMemConfigFile()
{
  m_memConfigFile.clear();
  // Read config file and env
  if (m_cfgParams.find(m_cfgVarMemConfig) != m_cfgParams.end()) {
    m_memConfigFile = m_cfgParams.at(m_cfgVarMemConfig);
  } else if (m_envParams.find(m_envVarMemConfig) != m_envParams.end()) {
    m_memConfigFile = m_envParams.at(m_envVarMemConfig);
  }
  if (!m_memConfigFile.empty()) {
    if (!std::filesystem::exists(m_memConfigFile)) {
      // Try to find it in the same directory of sim config file
      std::string configFilePath = pimUtils::getDirectoryPath(m_simConfigFile);
      if (std::filesystem::exists(configFilePath + "/" + m_memConfigFile)) {
        m_memConfigFile = configFilePath + "/" + m_memConfigFile;
      } else {
        std::cout << "PIM-Error: Cannot find memory config file: " << m_memConfigFile << std::endl;
        return false;
      }
    }

    // Determine memory protocol from memory config file. This is not sim config file.
    std::unordered_map<std::string, std::string> memParams = pimUtils::readParamsFromConfigFile(m_memConfigFile);
    if (memParams.find("protocol") != memParams.end()) {
      std::string protocol = memParams.at("protocol");
      if (protocol == "DDR3" || protocol == "DDR4" || protocol == "DDR5") {
        m_memoryProtocol = PIM_DEVICE_PROTOCOL_DDR;
      } else if (protocol == "LPDDR3" || protocol == "LPDDR4") {
        m_memoryProtocol = PIM_DEVICE_PROTOCOL_LPDDR;
      } else if (protocol == "HBM" || protocol == "HBM2") {
        m_memoryProtocol = PIM_DEVICE_PROTOCOL_HBM;
      } else {
        std::cout << "PIM-Error: Unknown protocol " << protocol << " in memory config file: " << m_memConfigFile << std::endl;
        return false;
      }
    } else {
      std::cout << "PIM-Error: Missing protocol parameter in memory config file: " << m_memConfigFile << std::endl;
      return false;
    }
  }
  return true;
}

//! @brief  Derive Params: A Specific PIM Memory Dimension
bool
pimSimConfig::deriveDimension(const std::string& cfgVar, const std::string& envVar, const unsigned apiVal, const unsigned defVal, unsigned& retVal)
{
  retVal = 0;

  bool hasVal = false;
  std::string valStr;

  // Check config file. Zero will be ignored
  valStr = pimUtils::getOptionalParam(m_cfgParams, cfgVar, hasVal);
  if (hasVal) {
    unsigned val = 0;
    bool ok = pimUtils::convertStringToUnsigned(valStr, val);
    if (!ok || val == 0) {
      std::cout << "PIM-Error: Incorrect config file parameter: " << cfgVar << "=" << valStr << std::endl;
      return false;
    }
    if (val > 0) {
      retVal = val;
      return true;
    }
  }

  // Check env var. Zero will be ignored
  valStr = pimUtils::getOptionalParam(m_envParams, envVar, hasVal);
  if (hasVal) {
    unsigned val = 0;
    bool ok = pimUtils::convertStringToUnsigned(valStr, val);
    if (!ok) {
      std::cout << "PIM-Error: Incorrect environment variable: " << envVar << "=" << valStr << std::endl;
      return false;
    }
    if (val > 0) {
      retVal = val;
      return true;
    }
  }

  // Check value from APIs
  retVal = (apiVal > 0) ? apiVal : defVal;
  return true;
}

//! @brief  Derive Params: PIM Memory Dimensions
bool
pimSimConfig::deriveDimensions(unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank, unsigned numRowPerSubarray, unsigned numColPerSubarray)
{
  bool ok = true;
  ok = ok & deriveDimension(m_cfgVarNumRanks, m_envVarNumRanks, numRanks, DEFAULT_NUM_RANKS, m_numRanks);
  ok = ok & deriveDimension(m_cfgVarNumBankPerRank, m_envVarNumBankPerRank, numBankPerRank, DEFAULT_NUM_BANK_PER_RANK, m_numBankPerRank);
  ok = ok & deriveDimension(m_cfgVarNumSubarrayPerBank, m_envVarNumSubarrayPerBank, numSubarrayPerBank, DEFAULT_NUM_SUBARRAY_PER_BANK, m_numSubarrayPerBank);
  ok = ok & deriveDimension(m_cfgVarNumRowPerSubarray, m_envVarNumRowPerSubarray, numRowPerSubarray, DEFAULT_NUM_ROW_PER_SUBARRAY, m_numRowPerSubarray);
  ok = ok & deriveDimension(m_cfgVarNumColPerSubarray, m_envVarNumColPerSubarray, numColPerSubarray, DEFAULT_NUM_COL_PER_SUBARRAY, m_numColPerSubarray);
  if (m_numRanks == 0 || m_numBankPerRank == 0 || m_numSubarrayPerBank == 0 || m_numRowPerSubarray == 0 || m_numColPerSubarray == 0) {
    std::cout << "PIM-Error: Memory dimension parameter cannot be 0" << std::endl;
    ok = false;
  }
  return ok;
}

//! @brief  Derive Params: Max number of threads
bool
pimSimConfig::deriveNumThreads()
{
  m_numThreads = 0;

  bool hasVal = false;
  std::string valStr;

  // Check config file
  if (m_numThreads == 0) {
    valStr = pimUtils::getOptionalParam(m_cfgParams, m_cfgVarMaxNumThreads, hasVal);
    if (hasVal) {
      unsigned val = 0;
      bool ok = pimUtils::convertStringToUnsigned(valStr, val);
      if (!ok) {
        std::cout << "PIM-Error: Incorrect config file parameter: " << m_cfgVarMaxNumThreads << "=" << valStr << std::endl;
        return false;
      }
      if (val > 0) {
        m_numThreads = val;
      }
    }
  }

  // Check env var. Zero will be ignored
  if (m_numThreads == 0) {
    valStr = pimUtils::getOptionalParam(m_envParams, m_envVarMaxNumThreads, hasVal);
    if (hasVal) {
      unsigned val = 0;
      bool ok = pimUtils::convertStringToUnsigned(valStr, val);
      if (!ok) {
        std::cout << "PIM-Error: Incorrect environment variable: " << m_envVarMaxNumThreads << "=" << valStr << std::endl;
        return false;
      }
      if (val > 0) {
        m_numThreads = val;
      }
    }
  }

  // Check hardware concurrency
  unsigned hwThreads = std::thread::hardware_concurrency();
  if (m_debug & pimSimConfig::DEBUG_PARAMS) {
    std::cout << "PIM-Debug: Maximum number of threads = " << m_numThreads << ", hardware concurrency = " << hwThreads << std::endl;
  }
  if (m_numThreads == 0) {
    m_numThreads = hwThreads;
  } else {
    m_numThreads = std::min(m_numThreads, hwThreads);
  }
  // Safety check
  if (m_numThreads < 1) {
    m_numThreads = 1;
  }
  return true;
}

//! @brief  Derive Params: Misc Env Vars
bool
pimSimConfig::deriveMiscEnvVars()
{
  bool hasVal = false;
  std::string valStr;

  m_analysisMode = false;
  valStr = pimUtils::getOptionalParam(m_envParams, m_envVarAnalysisMode, hasVal);
  if (hasVal) {
    if (valStr != "0" && valStr != "1") {
      std::cout << "PIM-Error: Incorrect environment variable: " << m_envVarAnalysisMode << "=" << valStr << std::endl;
      return false;
    }
    m_analysisMode = (valStr == "1");
  }
  if (m_analysisMode) {
    std::cout << "PIM-Warning: Running analysis only mode. Ignoring computation for fast performance and energy analysis." << std::endl;
  }

  return true;
}

//! @brief  Derive Params: Load balance - Distribute data evenly among parallel cores during allocation
bool
pimSimConfig::deriveLoadBalance()
{
  m_loadBalanced = false;

  // Check config file then env variable
  bool hasVal = false;
  std::string valStr = pimUtils::getOptionalParam(m_cfgParams, m_cfgVarLoadBalance, hasVal);
  if (hasVal) {
    if (valStr != "0" && valStr != "1") {
      std::cout << "PIM-Error: Incorrect config file parameter: " << m_cfgVarLoadBalance << "=" << valStr << std::endl;
      return false;
    }
    m_loadBalanced = (valStr == "1");
  } else {
    valStr = pimUtils::getOptionalParam(m_envParams, m_envVarLoadBalance, hasVal);
    if (hasVal) {
      if (valStr != "0" && valStr != "1") {
        std::cout << "PIM-Error: Incorrect environment variable: " << m_envVarLoadBalance << "=" << valStr << std::endl;
        return false;
      }
      m_loadBalanced = (valStr == "1");
    }
  }
  return true;
}

