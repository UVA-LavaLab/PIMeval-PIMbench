// File: pimSimConfig.h
// PIMeval Simulator - PIM Simulator Configurations
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_SIM_CONFIG_H
#define LAVA_PIM_SIM_CONFIG_H

#include "libpimeval.h"
#include <string>
#include <unordered_map>


//! @class  pimSimConfig
//! @brief  PIM simulator configurations
//!         Configure PIMeval library using configuration files and/or environment variables,
//!         after it is linked with PIM application executables
//!
//! Supported configuration file parameters:
//!   memory_config_file = <ini-file>            // memory config file, e.g., DDR4_8Gb_x16_3200.ini
//!   simulation_target = <PimDeviceEnum>        // simulation target, e.g., PIM_DEVICE_BITSIMD_V
//!   num_ranks = <int>                          // number of ranks
//!   num_bank_per_rank = <int>                  // number of banks per rank
//!   num_subarray_per_bank = <int>              // number of subarrays per bank
//!   num_row_per_subarray = <int>               // number of rows per subarray
//!   num_col_per_subarray = <int>               // number of columns per subarray
//!   max_num_threads = <int>                    // maximum number of threads used by simulation
//!   should_load_balance = <0|1>                // distribute data evenly among all cores
//!
//! Supported environment variables:
//!   PIMEVAL_SIM_CONFIG <abs-path/cfg-file>     // PIMeval config file, e.g., abs-path/PIMeval_BitSimdV.cfg
//!   PIMEVAL_MEM_CONFIG <abs-path/ini-file>     // memory config file, e.g., DDR4_8Gb_x16_3200.ini
//!   PIMEVAL_SIM_TARGET <PimDeviceEnum>         // simulation target, e.g., PIM_DEVICE_BITSIMD_V
//!   PIMEVAL_NUM_RANKS <int>                    // number of ranks
//!   PIMEVAL_NUM_BANK_PER_RANK <int>            // number of banks per rank
//!   PIMEVAL_NUM_SUBARRAY_PER_BANK <int>        // number of subarrays per bank
//!   PIMEVAL_NUM_ROW_PER_SUBARRAY <int>         // number of rows per subarray
//!   PIMEVAL_NUM_COL_PER_SUBARRAY <int>         // number of columns per subarray
//!   PIMEVAL_MAX_NUM_THREADS <int>              // maximum number of threads used by simulation
//!   PIMEVAL_ANALYSIS_MODE <0|1>                // PIMeval analysis mode
//!   PIMEVAL_DEBUG <int>                        // PIMeval debug flags (see enum pimDebugFlags)
//!   PIMEVAL_LOAD_BALANCE <0|1>                 // distribute data evenly among all cores
//!
//! Precedence rules (highest to lowest priority):
//! * Config file: Either from -c command-line argument or from PIMEVAL_SIM_CONFIG
//! * Environment variables
//! * Parameters from pimCreateDevice API or C++ macros
//!
//! About config file paths:
//! * Simulator config file
//!   - If passing it through -c command line argument, it's already a valid absolute or relative path
//!   - If using the PIMEVAL_SIM_CONFIG env variable, its value needs to be a valid absolute path
//! * Memory config file
//!   - If it is in the same directory as the simulator config file, specifying a file name is sufficient
//!   - Otherwise, use absolute path
//!
//! How to add a new configuration parameter:
//! * Define a key string below as m_cfgVar* and/or m_envVar*, and update the documentation above
//! * Define a member variable, and update the uninit and show function
//! * Add a private derive function (can reuse deriveMiscEnvVars if it's env only)
//! * Add a public getter function
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
  void uninit();
  void show() const;

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
  bool isAnalysisMode() const { return m_analysisMode; }
  unsigned getDebug() const { return m_debug; }
  bool isLoadBalanced() const { return m_loadBalanced; }

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
  bool deriveLoadBalance();

  bool parseConfigFromFile(const std::string& config, unsigned& numRanks, unsigned& numBankPerRank, unsigned& numSubarrayPerBank, unsigned& numRows, unsigned& numCols);

  // Default values if not specified by input parameters, config files, or env vars
  static constexpr int DEFAULT_NUM_RANKS = 1;
  static constexpr int DEFAULT_NUM_BANK_PER_RANK = 4;
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
  inline static const std::string m_envVarLoadBalance = "PIMEVAL_LOAD_BALANCE";

  // Config file parameter names
  inline static const std::string m_cfgVarMemConfig = "memory_config_file";
  inline static const std::string m_cfgVarSimTarget = "simulation_target";
  inline static const std::string m_cfgVarNumRanks = "num_ranks";
  inline static const std::string m_cfgVarNumBankPerRank = "num_bank_per_rank";
  inline static const std::string m_cfgVarNumSubarrayPerBank = "num_subarray_per_bank";
  inline static const std::string m_cfgVarNumRowPerSubarray = "num_row_per_subarray";
  inline static const std::string m_cfgVarNumColPerSubarray = "num_col_per_subarray";
  inline static const std::string m_cfgVarMaxNumThreads = "max_num_threads";
  inline static const std::string m_cfgVarLoadBalance = "should_load_balance";

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
  bool m_loadBalanced = false;

  // Store original parameters for extension purpose
  std::unordered_map<std::string, std::string> m_envParams;
  std::unordered_map<std::string, std::string> m_cfgParams;
};

#endif

