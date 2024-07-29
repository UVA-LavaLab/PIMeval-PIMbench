// File: pimParamsDram.cc
// PIMeval Simulator - DRAM parameters
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimParamsDram.h"


//! @brief  pimParamsDram ctor (based on DDR4_4Gb_x8_2666.ini from DRAMsim3)
pimParamsDram::pimParamsDram()
  : // [dram_structure]
    m_protocol("DDR4"),
    m_bankgroups(4),
    m_banksPerGroup(4),
    m_rows(32768),
    m_columns(1024),
    m_deviceWidth(8),
    m_BL(8),
    // [timing]
    m_tCK(0.75),
    m_AL(0),
    m_CL(19),
    m_CWL(14),
    m_tRCD(19),
    m_tRP(19),
    m_tRAS(43),
    m_tRFC(347),
    m_tRFC2(214),
    m_tRFC4(147),
    m_tREFI(10398),
    m_tRPRE(1),
    m_tWPRE(1),
    m_tRRD_S(4),
    m_tRRD_L(7),
    m_tWTR_S(4),
    m_tWTR_L(10),
    m_tFAW(28),
    m_tWR(20),
    m_tWR2(21),
    m_tRTP(10),
    m_tCCD_S(4),
    m_tCCD_L(7),
    m_tCKE(7),
    m_tCKESR(8),
    m_tXS(360),
    m_tXP(8),
    m_tRTRS(1),
    // [power]
    m_VDD(1.2),
    m_IDD0(65),
    m_IPP0(3.0),
    m_IDD2P(34),
    m_IDD2N(50),
    m_IDD3P(40),
    m_IDD3N(65),
    m_IDD4W(195),
    m_IDD4R(170),
    m_IDD5AB(175),
    m_IDD6x(20),
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
    m_epochPeriod(1333333),
    m_outputLevel(1)
{
}

//! @brief  pimParamsDram ctor with a config file
pimParamsDram::pimParamsDram(const std::string& configFile)
{
  // TODO: read .ini file
}

