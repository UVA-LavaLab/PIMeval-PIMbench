// File: pimUtils.cc
// PIMeval Simulator - Utilities
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimUtils.h"
#include "libpimeval.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <string>
#include <filesystem>
#include <cstdlib>
#include <cassert>
#include <stdexcept>


//! @brief  Convert PimStatus enum to string
std::string
pimUtils::pimStatusEnumToStr(PimStatus status)
{
  switch (status) {
  case PIM_ERROR: return "ERROR";
  case PIM_OK: return "OK";
  }
  return "Unknown";
}

//! @brief  Convert PimDeviceEnum to string
std::string 
pimUtils::pimDeviceEnumToStr(PimDeviceEnum deviceType) {
  auto it = enumToStrMap.find(deviceType);
  if (it != enumToStrMap.end()) {
    return it->second;
  }
  return "Unknown";
}

//! @brief  Convert string to PimDeviceEnum
PimDeviceEnum
pimUtils::strToPimDeviceEnum(const std::string& deviceTypeStr) {
  auto it = strToEnumMap.find(deviceTypeStr);
  if (it != strToEnumMap.end()) {
    return it->second;
  }
  return PIM_DEVICE_NONE;
}

//! @brief  Convert PimAllocEnum to string
std::string
pimUtils::pimAllocEnumToStr(PimAllocEnum allocType)
{
  switch (allocType) {
  case PIM_ALLOC_AUTO: return "PIM_ALLOC_AUTO";
  case PIM_ALLOC_V: return "PIM_ALLOC_V";
  case PIM_ALLOC_H: return "PIM_ALLOC_H";
  case PIM_ALLOC_V1: return "PIM_ALLOC_V1";
  case PIM_ALLOC_H1: return "PIM_ALLOC_H1";
  }
  return "Unknown";
}

//! @brief  Convert PimCopyEnum to string
std::string
pimUtils::pimCopyEnumToStr(PimCopyEnum copyType)
{
  switch (copyType) {
  case PIM_COPY_V: return "PIM_COPY_V";
  case PIM_COPY_H: return "PIM_COPY_H";
  }
  return "Unknown";
}

//! @brief  Convert PimDataType enum to string
std::string
pimUtils::pimDataTypeEnumToStr(PimDataType dataType)
{
  switch (dataType) {
  case PIM_BOOL: return "bool";
  case PIM_INT8: return "int8";
  case PIM_INT16: return "int16";
  case PIM_INT32: return "int32";
  case PIM_INT64: return "int64";
  case PIM_UINT8: return "uint8";
  case PIM_UINT16: return "uint16";
  case PIM_UINT32: return "uint32";
  case PIM_UINT64: return "uint64";
  case PIM_FP32: return "fp32";
  case PIM_FP16: return "fp16";
  case PIM_BF16: return "bf16";
  case PIM_FP8: return "fp8";
  }
  return "Unknown";
}

namespace pimUtils {
  //! @brief  Static definitions of bits of data types (see PimBitWidth)
  //! Notes:
  //! - BOOL: PIMeval requires host data to store one bool value per byte
  //! - FP16/BF16/FP8: PIMeval uses FP32 for functional simulation
  static const std::unordered_map<PimDataType, std::unordered_map<PimBitWidth, unsigned>> s_bitsOfDataType = {
    { PIM_BOOL, {{PimBitWidth::ACTUAL, 1}, {PimBitWidth::SIM, 1}, {PimBitWidth::HOST, 8}} },
    { PIM_INT8, {{PimBitWidth::ACTUAL, 8}, {PimBitWidth::SIM, 8}, {PimBitWidth::HOST, 8}} },
    { PIM_INT16, {{PimBitWidth::ACTUAL, 16}, {PimBitWidth::SIM, 16}, {PimBitWidth::HOST, 16}} },
    { PIM_INT32, {{PimBitWidth::ACTUAL, 32}, {PimBitWidth::SIM, 32}, {PimBitWidth::HOST, 32}} },
    { PIM_INT64, {{PimBitWidth::ACTUAL, 64}, {PimBitWidth::SIM, 64}, {PimBitWidth::HOST, 64}} },
    { PIM_UINT8, {{PimBitWidth::ACTUAL, 8}, {PimBitWidth::SIM, 8}, {PimBitWidth::HOST, 8}} },
    { PIM_UINT16, {{PimBitWidth::ACTUAL, 16}, {PimBitWidth::SIM, 16}, {PimBitWidth::HOST, 16}} },
    { PIM_UINT32, {{PimBitWidth::ACTUAL, 32}, {PimBitWidth::SIM, 32}, {PimBitWidth::HOST, 32}} },
    { PIM_UINT64, {{PimBitWidth::ACTUAL, 64}, {PimBitWidth::SIM, 64}, {PimBitWidth::HOST, 64}} },
    { PIM_FP32, {{PimBitWidth::ACTUAL, 32}, {PimBitWidth::SIM, 32}, {PimBitWidth::HOST, 32}} },
    { PIM_FP16, {{PimBitWidth::ACTUAL, 16}, {PimBitWidth::SIM, 32}, {PimBitWidth::HOST, 32}} },
    { PIM_BF16, {{PimBitWidth::ACTUAL, 16}, {PimBitWidth::SIM, 32}, {PimBitWidth::HOST, 32}} },
    { PIM_FP8, {{PimBitWidth::ACTUAL, 8}, {PimBitWidth::SIM, 32}, {PimBitWidth::HOST, 32}} },
  };
}

//! @brief  Get number of bits of a PIM data type
unsigned
pimUtils::getNumBitsOfDataType(PimDataType dataType, PimBitWidth bitWidthType)
{
  if (bitWidthType == PimBitWidth::ACTUAL || bitWidthType == PimBitWidth::SIM || bitWidthType == PimBitWidth::HOST) {
    auto it = pimUtils::s_bitsOfDataType.find(dataType);
    return it != s_bitsOfDataType.end() ? it->second.at(bitWidthType) : 0;
  }
  return 0;
}

//! @brief  Check if a PIM data type is signed integer
bool
pimUtils::isSigned(PimDataType dataType)
{
  return dataType == PIM_INT8 || dataType == PIM_INT16 || dataType == PIM_INT32 || dataType == PIM_INT64;
}

//! @brief  Check if a PIM data type is unsigned integer
bool
pimUtils::isUnsigned(PimDataType dataType)
{
  return dataType == PIM_BOOL || dataType == PIM_UINT8 || dataType == PIM_UINT16 || dataType == PIM_UINT32 || dataType == PIM_UINT64;
}

//! @brief  Check if a PIM data type is floating point
bool
pimUtils::isFP(PimDataType dataType)
{
  return dataType == PIM_FP32 || dataType == PIM_FP16 || dataType == PIM_BF16 || dataType == PIM_FP8;
}

//! @brief  Convert PimDeviceProtocolEnum to string
std::string
pimUtils::pimProtocolEnumToStr(PimDeviceProtocolEnum protocol)
{
  switch (protocol) {
    case PIM_DEVICE_PROTOCOL_DDR: return "DDR";
    case PIM_DEVICE_PROTOCOL_LPDDR: return "LPDDR";
    case PIM_DEVICE_PROTOCOL_HBM: return "HBM";
  }
  return "Unknown";
}

//! @brief  Get device data layout
PimDataLayout
pimUtils::getDeviceDataLayout(PimDeviceEnum deviceType)
{
  switch (deviceType) {
    case PIM_DEVICE_BITSIMD_V: return PimDataLayout::V;
    case PIM_DEVICE_BITSIMD_V_NAND: return PimDataLayout::V;
    case PIM_DEVICE_BITSIMD_V_MAJ: return PimDataLayout::V;
    case PIM_DEVICE_BITSIMD_V_AP: return PimDataLayout::V;
    case PIM_DEVICE_DRISA_NOR: return PimDataLayout::V;
    case PIM_DEVICE_DRISA_MIXED: return PimDataLayout::V;
    case PIM_DEVICE_SIMDRAM: return PimDataLayout::V;
    case PIM_DEVICE_BITSIMD_H: return PimDataLayout::H;
    case PIM_DEVICE_FULCRUM: return PimDataLayout::H;
    case PIM_DEVICE_BANK_LEVEL: return PimDataLayout::H;
    case PIM_DEVICE_AQUABOLT: return PimDataLayout::H;
    case PIM_DEVICE_AIM: return PimDataLayout::H;
    case PIM_FUNCTIONAL:
    case PIM_DEVICE_NONE: return PimDataLayout::UNKNOWN;
  }
  return PimDataLayout::UNKNOWN;
}

//! @brief  Thread pool ctor
pimUtils::threadPool::threadPool(size_t numThreads)
  : m_terminate(false),
    m_workersRemaining(0)
{
  // reserve one thread for main program
  for (size_t i = 1; i < numThreads; ++i) {
    m_threads.emplace_back([this] { workerThread(); });
  }
  std::printf("PIM-Info: Created thread pool with %lu threads.\n", m_threads.size());
}

//! @brief  Thread pool dtor
pimUtils::threadPool::~threadPool()
{
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_terminate = true;
  }
  m_cond.notify_all();
  for (auto& thread : m_threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}

//! @brief  Entry to process workers in MT
void
pimUtils::threadPool::doWork(const std::vector<pimUtils::threadWorker*>& workers)
{
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    for (auto& worker : workers) {
      m_workers.push(worker);
    }
    m_workersRemaining = workers.size();
  }
  m_cond.notify_all();

  // Wait for all workers to be done
  std::unique_lock<std::mutex> lock(m_mutex);
  m_cond.wait(lock, [this] { return m_workersRemaining == 0; });
}

//! @brief  Worker thread that process workers
void
pimUtils::threadPool::workerThread()
{
  while (true) {
    threadWorker* worker;
    {
      std::unique_lock<std::mutex> lock(m_mutex);
      m_cond.wait(lock, [this] { return m_terminate || !m_workers.empty(); });
      if (m_terminate && m_workers.empty()) {
        return;
      }
      worker = m_workers.front();
      m_workers.pop();
    }
    worker->execute();
    {
      std::unique_lock<std::mutex> lock(m_mutex);
      --m_workersRemaining;
    }
    m_cond.notify_all();
  }
}

//! @brief Helper function to trim from the start (left) of the string
std::string&
pimUtils::ltrim(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
    return !std::isspace(ch);
  }));
  return s;
 }

//! @brief Helper function to trim from the end (right) of the string
std::string&
pimUtils::rtrim(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
    return !std::isspace(ch);
  }).base(), s.end());
  return s;
}

//! @brief Function to trim from both ends
std::string&
pimUtils::trim(std::string& s) {
  return ltrim(rtrim(s));
}


//! @brief Reads the content of a file and stores it in a provided std::string reference
bool
pimUtils::readFileContent(const char* fileName, std::string& fileContent) {
    std::ifstream fileStream(fileName);

    if (!fileStream.is_open()) {
        std::cerr << "PIM-Error: Could not open the file: " << fileName << std::endl;
        return false;
    }

    std::stringstream buffer;
    buffer << fileStream.rdbuf();
    fileContent = buffer.str();
    return true;
}

//! @brief Retrieves the value of the specified environment variable.
bool
pimUtils::getEnvVar(const std::string &name, std::string &value) {
    const char* evnVal = std::getenv(name.c_str());
    if (evnVal == nullptr) {
        return false;
    } else {
        value = evnVal;
        return true;
    }
}

//! @brief Returns the values of each parameter in the config files.
std::string
pimUtils::getParam(const std::unordered_map<std::string, std::string>& params, const std::string& key) {
  auto it = params.find(key);
  if (it == params.end()) {
    throw std::invalid_argument(key);
  }
  return it->second;
}

//! @brief Returns the values of each parameter in the config files. Return empty string and false return status if the parameter value is not found
std::string 
pimUtils::getOptionalParam(const std::unordered_map<std::string, std::string>& params, const std::string& key, bool& returnStatus) {
  returnStatus = false;  
  auto it = params.find(key);
  if (it == params.end()) {
    return "";
  }
  returnStatus = true;
  return it->second;
} 

//! @brief Returns a substring from the beginning of the input string up to the first ';' character, or the entire string if ';' is not found
std::string 
pimUtils::removeAfterSemicolon(const std::string &input) {
    size_t pos = input.find(';');
    if (pos != std::string::npos) {
        return input.substr(0, pos);
    }
    return input;
}

//! @brief Returns the directory path of the input file.
std::string
pimUtils::getDirectoryPath(const std::string& filePath) {
    std::filesystem::path path(filePath);
    return path.parent_path().string() + "/";
}

//! @brief Convert a string to unsigned int. Return false and 0 if invalid
bool
pimUtils::convertStringToUnsigned(const std::string& str, unsigned& retVal)
{
  try {
    unsigned long val = std::stoul(str);
    if (val > std::numeric_limits<unsigned int>::max()) { // out of range
      throw std::out_of_range("Value exceeds unsigned int range");
    }
    retVal = static_cast<unsigned int>(val);
  } catch (const std::exception &e) {
    retVal = 0;
    return false;
  }
  return true;
}

//! @brief Given a config file path, read all parameters
std::unordered_map<std::string, std::string>
pimUtils::readParamsFromConfigFile(const std::string& configFilePath)
{
  std::unordered_map<std::string, std::string> params;
  if (configFilePath.empty()) {
    return params;
  }
  std::string contents;
  bool success = pimUtils::readFileContent(configFilePath.c_str(), contents);
  if (!success) {
    std::printf("PIM-Error: Failed to read config file %s\n", configFilePath.c_str());
    return params;
  }
  std::istringstream iss(contents);
  std::string line;
  while (std::getline(iss, line)) {
    line = pimUtils::removeAfterSemicolon(line);
    if (line.empty() || line[0] == '[') {
      continue;
    }
    size_t equalPos = line.find('=');
    if (equalPos != std::string::npos) {
      std::string key = line.substr(0, equalPos);
      std::string value = line.substr(equalPos + 1);
      params[pimUtils::trim(key)] = pimUtils::trim(value);
    }
  }
  return params;
}

//! @brief Read environment variables, given a list of env var names
std::unordered_map<std::string, std::string>
pimUtils::readParamsFromEnvVars(const std::vector<std::string>& envVarNames)
{
  std::unordered_map<std::string, std::string> params;
  for (const auto& envVar : envVarNames) {
    std::string val;
    bool hasEnv = pimUtils::getEnvVar(envVar, val);
    val = pimUtils::trim(val);
    if (hasEnv && !val.empty()) {
      params[envVar] = val;
    }
  }
  return params;
}

