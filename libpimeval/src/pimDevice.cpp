// File: pimDevice.cpp
// PIMeval Simulator - PIM Device
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimDevice.h"
#include "pimResMgr.h"
#include "pimSim.h"
#include "libpimeval.h"
#include "pimUtils.h"
#include <cstdio>
#include <deque>
#include <memory>
#include <cassert>
#include <sstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <locale>
#include <stdexcept>
#include <filesystem>

//! @brief  pimDevice ctor
pimDevice::pimDevice()
{
}

//! @brief  pimDevice dtor
pimDevice::~pimDevice()
{
}

//! @brief  Adjust config for modeling different simulation target with same inputs
bool
pimDevice::adjustConfigForSimTarget(unsigned& numRanks, unsigned& numBankPerRank, unsigned& numSubarrayPerBank, unsigned& numRows, unsigned& numCols)
{
  std::printf("PIM-Info: Config: #ranks = %u, #bankPerRank = %u, #subarrayPerBank = %u, #rowsPerSubarray = %u, #colsPerRow = %u\n",
              numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_NAND:
  case PIM_DEVICE_BITSIMD_V_MAJ:
  case PIM_DEVICE_BITSIMD_V_AP:
  case PIM_DEVICE_DRISA_NOR:
  case PIM_DEVICE_DRISA_MIXED:
  case PIM_DEVICE_SIMDRAM:
  case PIM_DEVICE_BITSIMD_H:
  case PIM_DEVICE_FULCRUM:
    std::printf("PIM-Info: Aggregate every two subarrays as a single core\n");
    if (numSubarrayPerBank % 2 != 0) {
      std::printf("PIM-Error: Please config even number of subarrays in each bank\n");
      return false;
    }
    numRows *= 2;
    numSubarrayPerBank /= 2;
    break;
  case PIM_DEVICE_BANK_LEVEL:
    std::printf("PIM-Info: Aggregate all subarrays within a bank as a single core\n");
    numRows *= numSubarrayPerBank;
    numSubarrayPerBank = 1;
    break;
  default:
    assert(0);
  }
  return true;
}

//! @brief  Config the simulation target
void
pimDevice::configSimTarget(PimDeviceEnum deviceType)
{
  m_simTarget = deviceType;
  // from 'make PIM_SIM_TARGET=...'
  #if defined(PIM_SIM_TARGET)
  if (deviceType == PIM_FUNCTIONAL) {
    m_simTarget = PIM_SIM_TARGET;
  }
  #endif

  // Default simulation target
  if (m_simTarget == PIM_FUNCTIONAL || m_simTarget == PIM_DEVICE_NONE) {
    m_simTarget = PIM_DEVICE_BITSIMD_V;
  }
}

//! @brief  If a PIM device uses vertical data layout
bool
pimDevice::isVLayoutDevice() const
{
  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V: return true;
  case PIM_DEVICE_BITSIMD_V_NAND: return true;
  case PIM_DEVICE_BITSIMD_V_MAJ: return true;
  case PIM_DEVICE_BITSIMD_V_AP: return true;
  case PIM_DEVICE_DRISA_NOR: return true;
  case PIM_DEVICE_DRISA_MIXED: return true;
  case PIM_DEVICE_SIMDRAM: return true;
  case PIM_DEVICE_BITSIMD_H: return false;
  case PIM_DEVICE_FULCRUM: return false;
  case PIM_DEVICE_BANK_LEVEL: return false;
  case PIM_DEVICE_NONE:
  case PIM_FUNCTIONAL:
  default:
    assert(0);
  }
  return false;
}

//! @brief  If a PIM device uses horizontal data layout
bool
pimDevice::isHLayoutDevice() const
{
  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V: return false;
  case PIM_DEVICE_BITSIMD_V_NAND: return false;
  case PIM_DEVICE_BITSIMD_V_MAJ: return false;
  case PIM_DEVICE_BITSIMD_V_AP: return false;
  case PIM_DEVICE_DRISA_NOR: return false;
  case PIM_DEVICE_DRISA_MIXED: return false;
  case PIM_DEVICE_SIMDRAM: return false;
  case PIM_DEVICE_BITSIMD_H: return true;
  case PIM_DEVICE_FULCRUM: return true;
  case PIM_DEVICE_BANK_LEVEL: return true;
  case PIM_DEVICE_NONE:
  case PIM_FUNCTIONAL:
  default:
    assert(0);
  }
  return false;
}

//! @brief  If a PIM device uses hybrid data layout
bool
pimDevice::isHybridLayoutDevice() const
{
  return false;
}

//! @brief  Init pim device, with input arguments
bool
pimDevice::init(PimDeviceEnum deviceType, unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank, unsigned numRows, unsigned numCols)
{
  assert(!m_isInit);

  // Determine simulation target
  m_deviceType = deviceType;
  m_simTarget = deviceType;
  if (deviceType == PIM_FUNCTIONAL) {
    // Read envirnment variable for the simulation target
    bool readSimTargetFromMakeArgument = false;
    std::printf("PIM-Info: Trying to read simulation target from envirnment variable %s\n", pimUtils::envVarPimEvalTarget);
    std::string pimEvalTarget;
    bool readEnvVarStatus = pimUtils::getEnvVar(pimUtils::envVarPimEvalTarget, pimEvalTarget);
    if (!readEnvVarStatus) {
      std::printf("PIM-Info: Could not read environment variable %s\n", pimUtils::envVarPimEvalTarget);
      readSimTargetFromMakeArgument = true;
    }
    m_simTarget = pimUtils::strToPimDeviceEnum(pimEvalTarget);
    if (m_simTarget == PIM_DEVICE_NONE) {
      std::printf("PIM-Warning: Invalid value %s for environment varialbe %s\n", pimEvalTarget.c_str(), pimUtils::envVarPimEvalTarget);
      readSimTargetFromMakeArgument = true;
    }
    if (readSimTargetFromMakeArgument){
      configSimTarget(m_deviceType);
    }
  }
  std::printf("PIM-Info: Current Device = %s, Simulation Target = %s\n",
              pimUtils::pimDeviceEnumToStr(m_deviceType).c_str(),
              pimUtils::pimDeviceEnumToStr(m_simTarget).c_str());

  // input params
  m_numRanks = numRanks;
  m_numBankPerRank = numBankPerRank;
  m_numSubarrayPerBank = numSubarrayPerBank;
  m_numRowPerSubarray = numRows;
  m_numColPerSubarray = numCols;

  if (adjustConfigForSimTarget(numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols)) {
    m_numCores = numRanks * numBankPerRank * numSubarrayPerBank;
    m_numRows = numRows;
    m_numCols = numCols;
  } else {
    return false;
  }

  m_isValid = (m_numRanks > 0 && m_numCores > 0 && m_numRows > 0 && m_numCols > 0);
  assert(m_numCols % 8 == 0);

  if (!m_isValid) {
    std::printf("PIM-Error: Incorrect device parameters: %u cores, %u rows, %u columns\n", m_numCores, m_numRows, m_numCols);
    return false;
  }

  m_resMgr = std::make_unique<pimResMgr>(this);
  const pimParamsDram& paramsDram = pimSim::get()->getParamsDram(); // created before pimDevice ctor
  pimPerfEnergyModelParams params(m_simTarget, m_numRanks, paramsDram);
  m_perfEnergyModel = pimPerfEnergyFactory::createPerfEnergyModel(params);

  // Disable simulated memory creation for functional simulation
  if (m_deviceType != PIM_FUNCTIONAL) {
    m_cores.resize(m_numCores, pimCore(m_numRows, m_numCols));
  }

  std::printf("PIM-Info: Created PIM device with %u cores, each with %u rows and %u columns.\n", m_numCores, m_numRows, m_numCols);

  m_isInit = true;
  return m_isValid;
}

//! @brief  Init pim device, with config file
bool
pimDevice::init(PimDeviceEnum deviceType, const char* configFileName)
{
  bool success = false;
  assert(!m_isInit);

  if (!configFileName) {
    std::printf("PIM-Error: Null PIM device config file name\n");
    return false;
  }
  if (!std::filesystem::exists(configFileName)) {
    std::printf("PIM-Error: Config file not found.\n");
    return false;
  }

  // Assign simulation target based on the input argument
  m_deviceType = deviceType;
  m_simTarget = deviceType;

  // Read file content
  unsigned numRanks;
  unsigned numBankPerRank;
  unsigned numSubarrayPerBank;
  unsigned numRows;
  unsigned numCols;

  std::string fileContent;
  success = pimUtils::readFileContent(configFileName, fileContent);
  if (!success) {
    std::printf("PIM-Error: Failed to read config file %s\n", configFileName);
    return false;
  }

  // input params
  success = parseConfigFromFile(fileContent, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols);
  if (!success) {
    std::printf("PIM-Error: Failed to parse config file %s\n", configFileName);
    return false;
  }

  std::printf("PIM-Info: Current Device = %s, Simulation Target = %s\n",
              pimUtils::pimDeviceEnumToStr(m_deviceType).c_str(),
              pimUtils::pimDeviceEnumToStr(m_simTarget).c_str());

  m_numRanks = numRanks;
  m_numBankPerRank = numBankPerRank;
  m_numSubarrayPerBank = numSubarrayPerBank;
  m_numRowPerSubarray = numRows;
  m_numColPerSubarray = numCols;

  if (adjustConfigForSimTarget(numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols)) {
    m_numCores = numRanks * numBankPerRank * numSubarrayPerBank;
    m_numRows = numRows;
    m_numCols = numCols;
  } else {
    return false;
  }

#ifdef DRAMSIM3_INTEG
  std::string configFile(configFileName);
  //TODO: DRAMSim3 requires an output directory but for our purpose we do not need it so sending empty string
  m_deviceMemory = new dramsim3::PIMCPU(configFile, "");
  m_deviceMemoryConfig = m_deviceMemory->getMemorySystem()->getConfig();
  u_int64_t rowsPerBank = m_deviceMemoryConfig->rows, columnPerRow = m_deviceMemoryConfig->columns * m_deviceMemoryConfig->device_width;

  // todo: adjust for sim target
  m_numRanks = 1;
  m_numCores = 16;
  m_numRows = rowsPerBank/m_numCores;
  m_numCols = columnPerRow;
#endif

  m_isValid = (m_numRanks > 0 && m_numCores > 0 && m_numRows > 0 && m_numCols > 0);
  assert(m_numCols % 8 == 0);

  if (!m_isValid) {
    std::printf("PIM-Error: Incorrect device parameters: %u cores, %u rows, %u columns\n", m_numCores, m_numRows, m_numCols);
    return false;
  }

  m_resMgr = std::make_unique<pimResMgr>(this);
  const pimParamsDram& paramsDram = pimSim::get()->getParamsDram(); // created before pimDevice ctor
  pimPerfEnergyModelParams params(m_simTarget, m_numRanks, paramsDram);
  m_perfEnergyModel = pimPerfEnergyFactory::createPerfEnergyModel(params);

  // Disable simulated memory creation for functional simulation
  if (m_deviceType != PIM_FUNCTIONAL) {
    m_cores.resize(m_numCores, pimCore(m_numRows, m_numCols));
  }

  std::printf("PIM-Info: Created PIM device with %u cores of %u rows and %u columns.\n", m_numCores, m_numRows, m_numCols);

  m_isInit = true;
  return m_isValid;
}

//! @brief Initilize the device config parameters by parsing the config file
bool
pimDevice::parseConfigFromFile(const std::string& config, unsigned& numRanks, unsigned& numBankPerRank, unsigned& numSubarrayPerBank, unsigned& numRows, unsigned& numCols)
{
  std::istringstream configStream(config);
  std::string line;
  std::unordered_map<std::string, std::string> params;

  while (std::getline(configStream, line)) {
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
  try {
    numRanks = std::stoi(pimUtils::getParam(params, "num_ranks"));
    numBankPerRank = std::stoi(pimUtils::getParam(params, "num_bank_per_rank"));
    numSubarrayPerBank = std::stoi(pimUtils::getParam(params, "num_subarray_per_bank"));
    numRows = std::stoi(pimUtils::getParam(params, "num_row_per_subarray"));
    numCols = std::stoi(pimUtils::getParam(params, "num_col_per_subarray"));
    if (m_deviceType == PIM_FUNCTIONAL) {
      m_simTarget = pimUtils::strToPimDeviceEnum(pimUtils::getParam(params, "simulation_target"));
      if (m_simTarget == PIM_DEVICE_NONE) {
        std::printf("PIM-Error: Invalid simulation target in config file\n");
        return false;
      }
    }
  } catch (const std::invalid_argument& e) {
    std::string missing = e.what();
    if (missing == "simulation_target") {
      // Read envirnment variable for the simulation target
      bool readSimTargetFromMakeArgument = false;
      std::printf("PIM-Info: Trying to read simulation target from envirnment variable %s\n", pimUtils::envVarPimEvalTarget);
      std::string pimEvalTarget;
      bool readEnvVarStatus = pimUtils::getEnvVar(pimUtils::envVarPimEvalTarget, pimEvalTarget);
      if (!readEnvVarStatus) {
        std::printf("PIM-Info: Could not read environment variable %s\n", pimUtils::envVarPimEvalTarget);
        readSimTargetFromMakeArgument = true;
      }
      m_simTarget = pimUtils::strToPimDeviceEnum(pimEvalTarget);
      if (m_simTarget == PIM_DEVICE_NONE) {
        std::printf("PIM-Warning: Invalid value %s for environment varialbe %s\n", pimEvalTarget.c_str(), pimUtils::envVarPimEvalTarget);
        readSimTargetFromMakeArgument = true;
      }
      if (readSimTargetFromMakeArgument){
        configSimTarget(m_deviceType);
      }
    } else {
      std::string errorMessage("PIM-Error: Missing or invalid parameter: ");
      errorMessage += missing;
      errorMessage += "\n";
      std::printf("%s", errorMessage.c_str());
      return false;
    }
  }
  return true;
}

//! @brief  Alloc a PIM object
PimObjId
pimDevice::pimAlloc(PimAllocEnum allocType, uint64_t numElements, unsigned bitsPerElement, PimDataType dataType)
{
  if (allocType == PIM_ALLOC_AUTO) {
    if (isVLayoutDevice()) {
      allocType = PIM_ALLOC_V;
    } else if (isHLayoutDevice()) {
      allocType = PIM_ALLOC_H;
    } else {
      assert(0);
    }
  }
  return m_resMgr->pimAlloc(allocType, numElements, bitsPerElement, dataType);
}

//! @brief  Alloc a PIM object assiciated to a reference object
PimObjId
pimDevice::pimAllocAssociated(unsigned bitsPerElement, PimObjId assocId, PimDataType dataType)
{
  return m_resMgr->pimAllocAssociated(bitsPerElement, assocId, dataType);
}

//! @brief  Free a PIM object
bool
pimDevice::pimFree(PimObjId obj)
{
  return m_resMgr->pimFree(obj);
}

//! @brief  Create an obj referencing to a range of an existing obj
PimObjId
pimDevice::pimCreateRangedRef(PimObjId refId, uint64_t idxBegin, uint64_t idxEnd)
{
  return m_resMgr->pimCreateRangedRef(refId, idxBegin, idxEnd);
}

//! @brief  Create an obj referencing to negation of an existing obj based on dual-contact memory cells
PimObjId
pimDevice::pimCreateDualContactRef(PimObjId refId)
{
  return m_resMgr->pimCreateDualContactRef(refId);
}

//! @brief  Copy data from host to PIM within a range
bool
pimDevice::pimCopyMainToDevice(void* src, PimObjId dest, uint64_t idxBegin, uint64_t idxEnd)
{
  PimCopyEnum copyType = m_resMgr->isHLayoutObj(dest) ? PIM_COPY_H : PIM_COPY_V;
  return pimCopyMainToDeviceWithType(copyType, src, dest, idxBegin, idxEnd);
}

//! @brief  Copy data from PIM to host within a range
bool
pimDevice::pimCopyDeviceToMain(PimObjId src, void* dest, uint64_t idxBegin, uint64_t idxEnd)
{
  PimCopyEnum copyType = m_resMgr->isHLayoutObj(src) ? PIM_COPY_H : PIM_COPY_V;
  return pimCopyDeviceToMainWithType(copyType, src, dest, idxBegin, idxEnd);
}

//! @brief  Copy data from host to PIM within a range
bool
pimDevice::pimCopyMainToDeviceWithType(PimCopyEnum copyType, void* src, PimObjId dest, uint64_t idxBegin, uint64_t idxEnd)
{
  std::unique_ptr<pimCmd> cmd =
    std::make_unique<pimCmdCopy>(PimCmdEnum::COPY_H2D, copyType, src, dest, idxBegin, idxEnd);
  return executeCmd(std::move(cmd));
}

//! @brief  Copy data from PIM to host within a range
bool
pimDevice::pimCopyDeviceToMainWithType(PimCopyEnum copyType, PimObjId src, void* dest, uint64_t idxBegin, uint64_t idxEnd)
{
  std::unique_ptr<pimCmd> cmd =
    std::make_unique<pimCmdCopy>(PimCmdEnum::COPY_D2H, copyType, src, dest, idxBegin, idxEnd);
  return executeCmd(std::move(cmd));
}

//! @brief  Copy data from PIM to PIM within a range
bool
pimDevice::pimCopyDeviceToDevice(PimObjId src, PimObjId dest, uint64_t idxBegin, uint64_t idxEnd)
{
  const pimObjInfo& obj = m_resMgr->getObjInfo(src);
  PimCopyEnum copyType = obj.isVLayout() ? PIM_COPY_V : PIM_COPY_H;
  std::unique_ptr<pimCmd> cmd =
    std::make_unique<pimCmdCopy>(PimCmdEnum::COPY_D2D, copyType, src, dest, idxBegin, idxEnd);
  return executeCmd(std::move(cmd));
}

//! @brief  Execute a PIM command
bool
pimDevice::executeCmd(std::unique_ptr<pimCmd> cmd)
{
  cmd->setDevice(this);
  bool ok = cmd->execute();

  return ok;
}

