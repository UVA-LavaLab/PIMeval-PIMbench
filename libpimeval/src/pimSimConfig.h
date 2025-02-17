// File: pimSimConfig.h
// PIMeval Simulator - PIM Simulator Configurations
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_SIM_CONFIG_H
#define LAVA_PIM_SIM_CONFIG_H

#include "libpimeval.h"
#include <string>                      // for string
#include <unordered_map>               // for unordered_map

//! @class  pimSimConfig
//! @brief  PIM simulator configurations
//! After the PIMeval library is linked to PIM application executables,
//! its behavior can be controlled using environment variables and/or configuration files.
//! This class is for managing the configuration parameters.
//!
//! Overriding rules (highest to lowest priority):
//! - Config files, either from -c command-line argument or PIMEVAL_SIM_CONFIG
//! - Other environment variables
//! - Parameters from pimCreateDevice API and C++ macros
//!
//! Supported environment parameters:
//!
//! PIMEVAL_SIM_CONFIG <full-path-to-simulator-config-file>
//! - Description: Specify a simulator config file
//! - Example:
//!     export PIMEVAL_SIM_CONFIG="<path-to-this-repo>/configs/PIMeval_BitSimdV.cfg"
//!
//! PIMEVAL_MEM_CONFIG <full-path-to-memory-config-ini-file>
//! - Description: Specify memory technology config file
//!   For backward compatility, the path can be relative path to PIMEVAL_SIM_CONFIG.
//! - Example:
//!     export PIMEVAL_MEM_CONFIG="<path-to-this-repo>/configs/DDR4_8Gb_x16_3200.ini"
//!     export PIMEVAL_MEM_CONFIG="DDR4_8Gb_x16_3200.ini"
//! - Equivalent config file parameter:
//!     memory_config_file = <ini-file>
//!
//! PIMEVAL_SIM_TARGET <PimDeviceEnum>
//! - Description: Specify simulation target
//! - Example:
//!     export PIMEVAL_SIM_TARGET="PIM_DEVICE_BITSIMD_V"
//! - Equivalent config file parameter:
//!     simulation_target = <PimDeviceEnum>
//!
//! PIMEVAL_NUM_RANKS <int>
//! PIMEVAL_NUM_BANK_PER_RANK <int>
//! PIMEVAL_NUM_SUBARRAY_PER_BANK <int>
//! PIMEVAL_NUM_ROW_PER_SUBARRAY <int>
//! PIMEVAL_NUM_COL_PER_SUBARRAY <int>
//! - Description: Memory dimension parameters
//! - Equivalent config file parameters:
//!     num_ranks = <int>
//!     num_bank_per_rank = <int>
//!     num_subarray_per_bank = <int>
//!     num_row_per_subarray = <int>
//!     num_col_per_subarray = <int>
//!
//! PIMEVAL_MAX_NUM_THREADS <int>
//! - Description: Specify max number of threads for simulation. If the number is negative,
//!   zero, or greater than hardware_concurrency, use hardware_concurrency
//! - Example:
//!     export PIMEVAL_MAX_NUM_THREADS=10
//! - Equivalent config file parameters:
//!     max_num_threads = <int>
//!
//! PIMEVAL_ANALYSIS_MODE <0|1>
//! - Description: Do fast performance analysis without functional computation. Data
//!   results will be incorrect in this mode. Not recommended for result dependent kernel
//! - Example:
//!     export PIMEVAL_ANALYSIS_MODE=1
//!
//! PIMEVAL_DEBUG <int>
//! - Description: Debug bit flags
//! - Example:
//!     export PIMEVAL_DEBUG=1
//!
//! TODO: Need future extension to support multi device or non-uniform device
//!
class pimSimConfig
{
public:
  pimSimConfig() {}
  ~pimSimConfig() {}

  // Update PIMeval simulation configuration parameters at device creation
  bool init(PimDeviceEnum deviceType, unsigned numRanks, unsigned numBankPerRank,
      unsigned numSubarrayPerBank, unsigned numRowPerSubarray, unsigned numColPerSubarray);
  bool init(PimDeviceEnum deviceType, const std::string& configFilePath);

  // Getters
  const std::string& getSimConfigFile() const { return m_simConfigFile; }
  const std::string& getMemConfigFile() const { return m_memConfigFile; }
  PimDeviceEnum getDeviceType() const { return m_deviceType; }
  PimDeviceEnum getSimTarget() const { return m_simTarget; }
  PimDeviceProtocolEnum getMemoryProtocol() const { return m_memoryProtocol; }
  unsigned getNumRanks() const { return m_numRanks; }
  unsigned getNumBankPerRank() const { return m_numBankPerRank; }
  unsigned getNumSubarrayPerBank() const { return m_numSubarrayPerBank; }
  unsigned getNumRowPerSubarray() const { return m_numRowPerSubarray; }
  unsigned getNumColPerSubarray() const { return m_numColPerSubarray; }
  unsigned getNumThreads() const { return m_numThreads; }
  bool getAnalysisMode() const { return m_analysisMode; }
  unsigned getDebug() const { return m_debug; }

  enum pimDebugFlags
  {
    DEBUG_PARAMS      = 0x0001,
    DEBUG_API_CALLS   = 0x0002,
    DEBUG_CMDS        = 0x0004,
    DEBUG_ALLOC       = 0x0008,
  };

private:
  bool deriveConfig(PimDeviceEnum deviceType,
      const std::string& configFilePath = "",
      unsigned numRanks = 0,
      unsigned numBankPerRank = 0,
      unsigned numSubarrayPerBank = 0,
      unsigned numRowPerSubarray = 0,
      unsigned numColPerSubarray = 0);

  bool deriveDebug();
  std::unordered_map<std::string, std::string> readEnvVars() const;
  bool deriveSimConfigFile(const std::string& configFilePath);
  std::unordered_map<std::string, std::string> readSimConfigFileParams() const;
  bool deriveDeviceType(PimDeviceEnum deviceType);
  bool deriveSimTarget();
  bool deriveMemConfigFile();
  bool deriveDimension(const std::string& envVar, const std::string& cfgVar, const unsigned apiVal, const unsigned defVal, unsigned& retVal);
  bool deriveDimensions(unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank, unsigned numRowPerSubarray, unsigned numColPerSubarray);
  bool deriveNumThreads();
  bool deriveMiscEnvVars();

  bool parseConfigFromFile(const std::string& config, unsigned& numRanks, unsigned& numBankPerRank, unsigned& numSubarrayPerBank, unsigned& numRows, unsigned& numCols);

  // Default values if not specified by input parameters, config files, or env vars
  static constexpr int DEFAULT_NUM_RANKS = 1;
  static constexpr int DEFAULT_NUM_BANK_PER_RANK = 128;
  static constexpr int DEFAULT_NUM_SUBARRAY_PER_BANK = 32;
  static constexpr int DEFAULT_NUM_ROW_PER_SUBARRAY = 1024;
  static constexpr int DEFAULT_NUM_COL_PER_SUBARRAY = 8192;
  static constexpr PimDeviceEnum DEFAULT_SIM_TARGET = PIM_DEVICE_BITSIMD_V;

  // Environment variable names
  inline static const std::string m_envVarSimConfig = "PIMEVAL_SIM_CONFIG";
  inline static const std::string m_envVarMemConfig = "PIMEVAL_MEM_CONFIG";
  inline static const std::string m_envVarSimTarget = "PIMEVAL_SIM_TARGET";
  inline static const std::string m_envVarNumRanks = "PIMEVAL_NUM_RANKS";
  inline static const std::string m_envVarNumBankPerRank = "PIMEVAL_NUM_BANK_PER_RANK";
  inline static const std::string m_envVarNumSubarrayPerBank = "PIMEVAL_NUM_SUBARRAY_PER_BANK";
  inline static const std::string m_envVarNumRowPerSubarray = "PIMEVAL_NUM_ROW_PER_SUBARRAY";
  inline static const std::string m_envVarNumColPerSubarray = "PIMEVAL_NUM_COL_PER_SUBARRAY";
  inline static const std::string m_envVarMaxNumThreads = "PIMEVAL_MAX_NUM_THREADS";
  inline static const std::string m_envVarAnalysisMode = "PIMEVAL_ANALYSIS_MODE";
  inline static const std::string m_envVarDebug = "PIMEVAL_DEBUG";

  // Config file parameter names
  inline static const std::string m_cfgVarMemConfig = "memory_config_file";
  inline static const std::string m_cfgVarSimTarget = "simulation_target";
  inline static const std::string m_cfgVarNumRanks = "num_ranks";
  inline static const std::string m_cfgVarNumBankPerRank = "num_bank_per_rank";
  inline static const std::string m_cfgVarNumSubarrayPerBank = "num_subarray_per_bank";
  inline static const std::string m_cfgVarNumRowPerSubarray = "num_row_per_subarray";
  inline static const std::string m_cfgVarNumColPerSubarray = "num_col_per_subarray";
  inline static const std::string m_cfgVarMaxNumThreads = "max_num_threads";

  // PIM sim env variables
  std::string m_simConfigFile;
  std::string m_memConfigFile;
  PimDeviceEnum m_deviceType = PIM_DEVICE_NONE;
  PimDeviceEnum m_simTarget = PIM_DEVICE_NONE;
  PimDeviceProtocolEnum m_memoryProtocol = PIM_DEVICE_PROTOCOL_DDR;
  unsigned m_numRanks = 0;
  unsigned m_numBankPerRank = 0;
  unsigned m_numSubarrayPerBank = 0;
  unsigned m_numRowPerSubarray = 0;
  unsigned m_numColPerSubarray = 0;
  unsigned m_numThreads = 0;
  bool m_analysisMode = false;
  unsigned m_debug = 0;

  // Store original parameters for extension purpose
  std::unordered_map<std::string, std::string> m_envParams;
  std::unordered_map<std::string, std::string> m_cfgParams;
};

#endif

