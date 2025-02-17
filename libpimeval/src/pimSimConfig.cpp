// File: pimSimConfig.cpp
// PIMeval Simulator - PIM Simulator Configurations
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimSimConfig.h"
#include "pimUtils.h"
#include <sstream>
#include <unordered_map>
#include <iostream>
#include <filesystem>


//! @brief  Update PIMeval simulation configuration parameters at device creation
bool
pimSimConfig::updateConfig(PimDeviceEnum deviceType,
    unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank,
    unsigned numRowPerSubarray, unsigned numColPerSubarray)
{
  return deriveConfig(deviceType, "",
                      numRanks, numBankPerRank, numSubarrayPerBank,
                      numRowPerSubarray, numColPerSubarray);
}

//! @brief  Update PIMeval simulation configuration parameters at device creation
bool
pimSimConfig::updateConfig(PimDeviceEnum deviceType, const std::string& configFilePath)
{
  return deriveConfig(deviceType, configFilePath);
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
  ok = ok & deriveDebug();
  ok = ok & deriveDeviceType(deviceType);
  ok = ok & deriveSimTarget();
  ok = ok & deriveMemConfigFile();
  ok = ok & deriveDimensions(numRanks, numBankPerRank, numSubarrayPerBank, numRowPerSubarray, numColPerSubarray);
  ok = ok & deriveMaxNumThreads();

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
  if (m_debug > 0) {
    std::cout << "PIM-Debug: Debug flags = 0x" << std::hex << m_debug << std::endl;
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
    if (m_debug & pimSimConfig::DEBUG_PARAMS) {
      std::cout << "PIM-Debug: Simulator config file: " << m_simConfigFile << std::endl;
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
  if (m_debug & pimSimConfig::DEBUG_PARAMS) {
    std::cout << "PIM-Debug: Device type = " << pimUtils::pimDeviceEnumToStr(m_deviceType) << std::endl;
  }
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

  if (m_debug & pimSimConfig::DEBUG_PARAMS) {
    std::cout << "PIM-Debug: Simulation target = " << pimUtils::pimDeviceEnumToStr(m_simTarget) << std::endl;
  }
  return true;
}

//! @brief  Derive Params: Memory Config File
bool
pimSimConfig::deriveMemConfigFile()
{
  m_memConfigFile.clear();
  if (m_cfgParams.find(m_cfgVarMemConfig) != m_cfgParams.end()) {
    m_memConfigFile = m_cfgParams.at(m_cfgVarMemConfig);
  } else if (m_envParams.find(m_envVarMemConfig) != m_envParams.end()) {
    m_memConfigFile = m_envParams.at(m_envVarMemConfig);
  }
  if (!m_memConfigFile.empty()) {
    if (!std::filesystem::exists(m_memConfigFile)) {
      std::cout << "PIM-Error: Cannot find memory config file: " << m_memConfigFile << std::endl;
      return false;
    }
    if (m_debug & pimSimConfig::DEBUG_PARAMS) {
      std::cout << "PIM-Debug: Memory config file: " << m_memConfigFile << std::endl;
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
  if (ok && (m_debug & pimSimConfig::DEBUG_PARAMS)) {
    std::cout << "PIM-Debug: Num ranks: " << m_numRanks << std::endl;
  }
  ok = ok & deriveDimension(m_cfgVarNumBankPerRank, m_envVarNumBankPerRank, numBankPerRank, DEFAULT_NUM_BANK_PER_RANK, m_numBankPerRank);
  if (ok && (m_debug & pimSimConfig::DEBUG_PARAMS)) {
    std::cout << "PIM-Debug: Num banks per rank: " << m_numBankPerRank << std::endl;
  }
  ok = ok & deriveDimension(m_cfgVarNumSubarrayPerBank, m_envVarNumSubarrayPerBank, numSubarrayPerBank, DEFAULT_NUM_SUBARRAY_PER_BANK, m_numSubarrayPerBank);
  if (ok && (m_debug & pimSimConfig::DEBUG_PARAMS)) {
    std::cout << "PIM-Debug: Num subarrays per bank: " << m_numSubarrayPerBank << std::endl;
  }
  ok = ok & deriveDimension(m_cfgVarNumRowPerSubarray, m_envVarNumRowPerSubarray, numRowPerSubarray, DEFAULT_NUM_ROW_PER_SUBARRAY, m_numRowPerSubarray);
  if (ok && (m_debug & pimSimConfig::DEBUG_PARAMS)) {
    std::cout << "PIM-Debug: Num rows per subarray: " << m_numRowPerSubarray << std::endl;
  }
  ok = ok & deriveDimension(m_cfgVarNumColPerSubarray, m_envVarNumColPerSubarray, numColPerSubarray, DEFAULT_NUM_COL_PER_SUBARRAY, m_numColPerSubarray);
  if (ok && (m_debug & pimSimConfig::DEBUG_PARAMS)) {
    std::cout << "PIM-Debug: Num columns per subarray: " << m_numColPerSubarray << std::endl;
  }
  return ok;
}

//! @brief  Derive Params: Max number of threads
bool
pimSimConfig::deriveMaxNumThreads()
{
  m_maxNumThreads = 0;

  bool hasVal = false;
  std::string valStr;

  // Check config file
  if (m_maxNumThreads == 0) {
    valStr = pimUtils::getOptionalParam(m_cfgParams, m_cfgVarMaxNumThreads, hasVal);
    if (hasVal) {
      unsigned val = 0;
      bool ok = pimUtils::convertStringToUnsigned(valStr, val);
      if (!ok) {
        std::cout << "PIM-Error: Incorrect config file parameter: " << m_cfgVarMaxNumThreads << "=" << valStr << std::endl;
        return false;
      }
      if (val > 0) {
        m_maxNumThreads = val;
      }
    }
  }

  // Check env var. Zero will be ignored
  if (m_maxNumThreads == 0) {
    valStr = pimUtils::getOptionalParam(m_envParams, m_envVarMaxNumThreads, hasVal);
    if (hasVal) {
      unsigned val = 0;
      bool ok = pimUtils::convertStringToUnsigned(valStr, val);
      if (!ok) {
        std::cout << "PIM-Error: Incorrect environment variable: " << m_envVarMaxNumThreads << "=" << valStr << std::endl;
        return false;
      }
      if (val > 0) {
        m_maxNumThreads = val;
      }
    }
  }

  // Check hardware concurrency
  unsigned hwThreads = std::thread::hardware_concurrency();
  if (m_maxNumThreads == 0) {
    m_maxNumThreads = hwThreads;
  } else {
    m_maxNumThreads = std::min(m_maxNumThreads, hwThreads);
  }
  // Safety check
  if (m_maxNumThreads < 1) {
    m_maxNumThreads = 1;
  }
  if (m_debug & pimSimConfig::DEBUG_PARAMS) {
    std::cout << "PIM-Debug: Max number of threads = " << m_maxNumThreads << ", hardware concurrency = " << hwThreads << std::endl;
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

  return true;
}
