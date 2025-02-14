// File: pimParamsDDRDram.cc
// PIMeval Simulator - DDR DRAM parameters
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimParamsHBMDram.h"
#include "pimUtils.h"
#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <locale>
#include <stdexcept>

//! @brief  pimParamsHBMDram ctor (based on HBM2_4Gb_x128.ini from DRAMsim3)
pimParamsHBMDram::pimParamsHBMDram()
  : // [dram_structure]
  m_protocol("HBM"),
  m_bankgroups(4),
  m_banksPerGroup(4),
  m_rows(16384),
  m_columns(64),
  m_deviceWidth(128),
  m_BL(4),
  m_numDies(4),

  // [timing]
  m_tCK(1.0),
  m_CL(14),
  m_CWL(4),
  m_tRCDRD(14),
  m_tRCDWR(14),
  m_tRP(14),
  m_tRAS(34),
  m_tRFC(260),
  m_tREFI(3900),
  m_tREFIb(128),
  m_tRPRE(1),
  m_tWPRE(1),
  m_tRRD_S(4),
  m_tRRD_L(6),
  m_tWTR_S(6),
  m_tWTR_L(8),
  m_tFAW(30),
  m_tWR(16),
  m_tCCD_S(1),
  m_tCCD_L(2),
  m_tXS(268),
  m_tCKE(8),
  m_tCKSRE(10),
  m_tXP(8),
  m_tRTP_L(6),
  m_tRTP_S(4),

  // [power]
  m_VDD(1.2),
  m_IDD0(65),
  m_IDD2P(28),
  m_IDD2N(40),
  m_IDD3P(40),
  m_IDD3N(55),
  m_IDD4W(500),
  m_IDD4R(390),
  m_IDD5AB(250),
  m_IDD6x(31),

  // [system]
  m_channelSize(512),
  m_channels(8),
  m_busWidth(128),
  m_addressMapping("rorabgbachco"),
  m_queueStructure("PER_BANK"),
  m_rowBufPolicy("OPEN_PAGE"),
  m_cmdQueueSize(8),
  m_transQueueSize(32),
  m_unifiedQueue(false),

  // [other]
  m_epochPeriod(1000000),
  m_outputLevel(1)
{
}

//! @brief  pimParamsHBMDram ctor with a config file
pimParamsHBMDram::pimParamsHBMDram(std::unordered_map<std::string, std::string> params)
{
  try {
    m_protocol = pimUtils::getParam(params, "protocol");
    m_bankgroups = std::stoi(pimUtils::getParam(params, "bankgroups"));
    m_banksPerGroup = std::stoi(pimUtils::getParam(params, "banks_per_group"));
    m_rows = std::stoi(pimUtils::getParam(params, "rows"));
    m_columns = std::stoi(pimUtils::getParam(params, "columns"));
    m_deviceWidth = std::stoi(pimUtils::getParam(params, "device_width"));
    m_BL = std::stoi(pimUtils::getParam(params, "BL"));
    m_numDies = std::stoi(pimUtils::getParam(params, "num_dies"));

    m_tCK = std::stod(pimUtils::getParam(params, "tCK"));
    m_CL = std::stoi(pimUtils::getParam(params, "CL"));
    m_CWL = std::stoi(pimUtils::getParam(params, "CWL"));
    m_tRCDRD = std::stoi(pimUtils::getParam(params, "tRCDRD"));
    m_tRCDWR = std::stoi(pimUtils::getParam(params, "tRCDWR"));
    m_tRP = std::stoi(pimUtils::getParam(params, "tRP"));
    m_tRAS = std::stoi(pimUtils::getParam(params, "tRAS"));
    m_tRFC = std::stoi(pimUtils::getParam(params, "tRFC"));
    m_tREFI = std::stoi(pimUtils::getParam(params, "tREFI"));
    m_tREFIb = std::stoi(pimUtils::getParam(params, "tREFIb"));
    m_tRPRE = std::stoi(pimUtils::getParam(params, "tRPRE"));
    m_tWPRE = std::stoi(pimUtils::getParam(params, "tWPRE"));
    m_tRRD_S = std::stoi(pimUtils::getParam(params, "tRRD_S"));
    m_tRRD_L = std::stoi(pimUtils::getParam(params, "tRRD_L"));
    m_tWTR_S = std::stoi(pimUtils::getParam(params, "tWTR_S"));
    m_tWTR_L = std::stoi(pimUtils::getParam(params, "tWTR_L"));
    m_tFAW = std::stoi(pimUtils::getParam(params, "tFAW"));
    m_tWR = std::stoi(pimUtils::getParam(params, "tWR"));
    m_tCCD_S = std::stoi(pimUtils::getParam(params, "tCCD_S"));
    m_tCCD_L = std::stoi(pimUtils::getParam(params, "tCCD_L"));
    m_tXS = std::stoi(pimUtils::getParam(params, "tXS"));
    m_tCKE = std::stoi(pimUtils::getParam(params, "tCKE"));
    m_tCKSRE = std::stoi(pimUtils::getParam(params, "tCKSRE"));
    m_tXP = std::stoi(pimUtils::getParam(params, "tXP"));
    m_tRTP_L = std::stoi(pimUtils::getParam(params, "tRTP_L"));
    m_tRTP_S = std::stoi(pimUtils::getParam(params, "tRTP_S"));

    m_VDD = std::stod(pimUtils::getParam(params, "VDD"));
    m_IDD0 = std::stoi(pimUtils::getParam(params, "IDD0"));
    m_IDD2P = std::stoi(pimUtils::getParam(params, "IDD2P"));
    m_IDD2N = std::stoi(pimUtils::getParam(params, "IDD2N"));
    m_IDD3P = std::stoi(pimUtils::getParam(params, "IDD3P"));
    m_IDD3N = std::stoi(pimUtils::getParam(params, "IDD3N"));
    m_IDD4W = std::stoi(pimUtils::getParam(params, "IDD4W"));
    m_IDD4R = std::stoi(pimUtils::getParam(params, "IDD4R"));
    m_IDD5AB = std::stoi(pimUtils::getParam(params, "IDD5AB"));
    m_IDD6x = std::stoi(pimUtils::getParam(params, "IDD6x"));

    m_channelSize = std::stoi(pimUtils::getParam(params, "channel_size"));
    m_channels = std::stoi(pimUtils::getParam(params, "channels"));
    m_busWidth = std::stoi(pimUtils::getParam(params, "bus_width"));
    m_addressMapping = pimUtils::getParam(params, "address_mapping");
    m_queueStructure = pimUtils::getParam(params, "queue_structure");
    m_rowBufPolicy = pimUtils::getParam(params, "row_buf_policy");
    m_cmdQueueSize = std::stoi(pimUtils::getParam(params, "cmd_queue_size"));
    m_transQueueSize = std::stoi(pimUtils::getParam(params, "trans_queue_size"));
    m_unifiedQueue = pimUtils::getParam(params, "address_mapping") == "False" ? false : true;

    m_epochPeriod = std::stoi(pimUtils::getParam(params, "epoch_period"));
    m_outputLevel = std::stoi(pimUtils::getParam(params, "output_level"));
  } catch (const std::invalid_argument& e) {
    std::string errorMessage("PIM-Error: Missing or invalid parameter: ");
    errorMessage += e.what();
    errorMessage += "\n";
    throw std::invalid_argument(errorMessage);
  }
}

