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
  case PIM_INT8: return "int8";
  case PIM_INT16: return "int16";
  case PIM_INT32: return "int32";
  case PIM_INT64: return "int64";
  case PIM_UINT8: return "uint8";
  case PIM_UINT16: return "uint16";
  case PIM_UINT32: return "uint32";
  case PIM_UINT64: return "uint64";
  case PIM_FP32: return "fp32";
  }
  return "Unknown";
}

//! @brief  Get number of bits of a PIM data type
unsigned
pimUtils::getNumBitsOfDataType(PimDataType dataType)
{
  switch (dataType) {
  case PIM_INT8: return 8;
  case PIM_INT16: return 16;
  case PIM_INT32: return 32;
  case PIM_INT64: return 64;
  case PIM_UINT8: return 8;
  case PIM_UINT16: return 16;
  case PIM_UINT32: return 32;
  case PIM_UINT64: return 64;
  case PIM_FP32: return 32;
  default:
    assert(0);
  }
  return 0;
}

//! @brief  Read bits from host
std::vector<bool>
pimUtils::readBitsFromHost(void* src, uint64_t numElements, unsigned bitsPerElement)
{
  std::vector<bool> bits;
  unsigned char* bytePtr = static_cast<unsigned char*>(src);

  for (uint64_t i = 0; i < numElements * bitsPerElement; i += 8) {
    uint64_t byteIdx = i / 8;
    unsigned char byteVal = *(bytePtr + byteIdx);
    for (int j = 0; j < 8; ++j) {
      bits.push_back(byteVal & 1);
      byteVal = byteVal >> 1;
    }
  }

  return bits;
}

//! @brief  Write bits to host
bool
pimUtils::writeBitsToHost(void* dest, const std::vector<bool>& bits)
{
  unsigned char* bytePtr = static_cast<unsigned char*>(dest);
  uint64_t byteIdx = 0;

  for (uint64_t i = 0; i < bits.size(); i += 8) {
    unsigned char byteVal = 0;
    for (int j = 7; j >= 0; --j) {
      byteVal = byteVal << 1;
      byteVal |= bits[i + j];
    }
    *(bytePtr + byteIdx) = byteVal;
    byteIdx++;
  }

  return true;
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
