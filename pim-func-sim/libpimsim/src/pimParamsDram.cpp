// File: pimParamsDram.cc
// PIM Functional Simulator - DRAM parameters
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimParamsDram.h"


//! @brief  pimParamsDram ctor (based on DDR4_8Gb_x16_3200.ini from DRAMsim3)
pimParamsDram::pimParamsDram()
  : // [dram_structure]
    m_protocol("DDR4"),
    m_bankgroups(2),
    m_banksPerGroup(4),
    m_rows(65536),
    m_columns(1024),
    m_deviceWidth(16),
    m_BL(8),
    // [timing]
    m_tCK(0.63),
    m_AL(0),
    m_CL(22),
    m_CWL(16),
    m_tRCD(22),
    m_tRP(22),
    m_tRAS(52),
    m_tRFC(560),
    m_tRFC2(416),
    m_tRFC4(256),
    m_tREFI(12480),
    m_tRPRE(1),
    m_tWPRE(1),
    m_tRRD_S(9),
    m_tRRD_L(11),
    m_tWTR_S(4),
    m_tWTR_L(12),
    m_tFAW(48),
    m_tWR(24),
    m_tWR2(25),
    m_tRTP(12),
    m_tCCD_S(4),
    m_tCCD_L(8),
    m_tCKE(8),
    m_tCKESR(9),
    m_tXS(576),
    m_tXP(10),
    m_tRTRS(1),
    // [power]
    m_VDD(1.2),
    m_IDD0(95),
    m_IPP0(4.0),
    m_IDD2P(25),
    m_IDD2N(37),
    m_IDD3P(47),
    m_IDD3N(56),
    m_IDD4W(278),
    m_IDD4R(302),
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
    m_epochPeriod(1587301),
    m_outputLevel(1)
{
}

//! @brief  pimParamsDram ctor with a config file
pimParamsDram::pimParamsDram(const std::string& configFile)
{
  // TODO: read .ini file
}

