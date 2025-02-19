// File: pimParamsLPDDRDram.cc
// PIMeval Simulator - LPDDR DRAM parameters
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimParamsLPDDRDram.h"
#include "pimUtils.h"
#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <locale>
#include <stdexcept>

//! @brief  pimParamsLPDDRDram ctor (based on LPDDR4_8Gb_x16_2400.ini from DRAMsim3)
pimParamsLPDDRDram::pimParamsLPDDRDram()
  : // [dram_structure]
    m_protocol("LPDDR4"),
    m_bankgroups(2),
    m_banksPerGroup(4),
    m_rows(65536),
    m_columns(1024),
    m_deviceWidth(16),
    m_BL(16),

    // [timing]
    m_tCK(0.83),
    m_AL(0),
    m_CL(17),
    m_CWL(14),
    m_tRCD(15),
    m_tRP(15),
    m_tRAS(32),
    m_tRFC(392),
    m_tRFC2(268),
    m_tRFC4(172),
    m_tREFI(8660),
    m_tRPRE(1),
    m_tWPRE(1),
    m_tRRD_S(8),
    m_tRRD_L(8),
    m_tWTR_S(8),
    m_tWTR_L(16),
    m_tFAW(32),
    m_tWR(30),
    m_tWR2(32),
    m_tRTP(12),
    m_tCCD_S(4),
    m_tCCD_L(6),
    m_tCKE(6),
    m_tCKESR(7),
    m_tXS(360),
    m_tXP(8),
    m_tRTRS(1),
    m_tPPD(2),
    // [power]
    m_VDD(1.2),
    m_IDD0(80),
    m_IPP0(4.0),
    m_IDD2P(25),
    m_IDD2N(34),
    m_IDD3P(41),
    m_IDD3N(47),
    m_IDD4W(228),
    m_IDD4R(243),
    m_IDD5AB(280),
    m_IDD6x(30),
    // [system]
    m_channelSize(8192),
    m_channels(1),
    m_busWidth(64),
    m_addressMapping("rochrababgco"),
    m_queueStructure("PER_BANK"),
    m_refreshPolicy("RANK_LEVEL_STAGGERED"),
    m_rowBufPolicy("OPEN_PAGE"),
    m_cmdQueueSize(8),
    m_transQueueSize(32),
    // [other]
    m_epochPeriod(1204819),
    m_outputLevel(1)
{
}

//! @brief  pimParamsLPDDRDram ctor with a config file
pimParamsLPDDRDram::pimParamsLPDDRDram(std::unordered_map<std::string, std::string> params)
{
  try {
    m_protocol = pimUtils::getParam(params, "protocol");
    m_bankgroups = std::stoi(pimUtils::getParam(params, "bankgroups"));
    m_banksPerGroup = std::stoi(pimUtils::getParam(params, "banks_per_group"));
    m_rows = std::stoi(pimUtils::getParam(params, "rows"));
    m_columns = std::stoi(pimUtils::getParam(params, "columns"));
    m_deviceWidth = std::stoi(pimUtils::getParam(params, "device_width"));
    m_BL = std::stoi(pimUtils::getParam(params, "BL"));

    m_tCK = std::stod(pimUtils::getParam(params, "tCK"));
    m_AL = std::stoi(pimUtils::getParam(params, "AL"));
    m_CL = std::stoi(pimUtils::getParam(params, "CL"));
    m_CWL = std::stoi(pimUtils::getParam(params, "CWL"));
    m_tRCD = std::stoi(pimUtils::getParam(params, "tRCD"));
    m_tRP = std::stoi(pimUtils::getParam(params, "tRP"));
    m_tRAS = std::stoi(pimUtils::getParam(params, "tRAS"));
    m_tRFC = std::stoi(pimUtils::getParam(params, "tRFC"));
    m_tRFC2 = std::stoi(pimUtils::getParam(params, "tRFC2"));
    m_tRFC4 = std::stoi(pimUtils::getParam(params, "tRFC4"));
    m_tREFI = std::stoi(pimUtils::getParam(params, "tREFI"));
    m_tRPRE = std::stoi(pimUtils::getParam(params, "tRPRE"));
    m_tWPRE = std::stoi(pimUtils::getParam(params, "tWPRE"));
    m_tRRD_S = std::stoi(pimUtils::getParam(params, "tRRD_S"));
    m_tRRD_L = std::stoi(pimUtils::getParam(params, "tRRD_L"));
    m_tWTR_S = std::stoi(pimUtils::getParam(params, "tWTR_S"));
    m_tWTR_L = std::stoi(pimUtils::getParam(params, "tWTR_L"));
    m_tFAW = std::stoi(pimUtils::getParam(params, "tFAW"));
    m_tWR = std::stoi(pimUtils::getParam(params, "tWR"));
    m_tWR2 = std::stoi(pimUtils::getParam(params, "tWR2"));
    m_tRTP = std::stoi(pimUtils::getParam(params, "tRTP"));
    m_tCCD_S = std::stoi(pimUtils::getParam(params, "tCCD_S"));
    m_tCCD_L = std::stoi(pimUtils::getParam(params, "tCCD_L"));
    m_tCKE = std::stoi(pimUtils::getParam(params, "tCKE"));
    m_tCKESR = std::stoi(pimUtils::getParam(params, "tCKESR"));
    m_tXS = std::stoi(pimUtils::getParam(params, "tXS"));
    m_tXP = std::stoi(pimUtils::getParam(params, "tXP"));
    m_tRTRS = std::stoi(pimUtils::getParam(params, "tRTRS"));
    m_tPPD = std::stoi(pimUtils::getParam(params, "tPPD"));

    m_VDD = std::stod(pimUtils::getParam(params, "VDD"));
    m_IDD0 = std::stoi(pimUtils::getParam(params, "IDD0"));
    m_IPP0 = std::stod(pimUtils::getParam(params, "IPP0"));
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
    m_refreshPolicy = pimUtils::getParam(params, "refresh_policy");
    m_rowBufPolicy = pimUtils::getParam(params, "row_buf_policy");
    m_cmdQueueSize = std::stoi(pimUtils::getParam(params, "cmd_queue_size"));
    m_transQueueSize = std::stoi(pimUtils::getParam(params, "trans_queue_size"));

    m_epochPeriod = std::stoi(pimUtils::getParam(params, "epoch_period"));
    m_outputLevel = std::stoi(pimUtils::getParam(params, "output_level"));
  } catch (const std::invalid_argument& e) {
    std::string errorMessage("PIM-Error: Missing or invalid parameter: ");
    errorMessage += e.what();
    errorMessage += "\n";
    throw std::invalid_argument(errorMessage);
  }
}

