// File: pimParamsDram.h
// PIM Functional Simulator - DRAM parameters
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_PARAMS_DRAM_H
#define LAVA_PIM_PARAMS_DRAM_H

#include <string>

//! @class  pimParamsDram
//! @brief  DRAM parameters (DRAMsim3 compatible)
class pimParamsDram
{
public:
  pimParamsDram();
  pimParamsDram(const std::string& configFile);
  ~pimParamsDram() {}

  float getNsRowRead() const { return m_tCK * (m_tRCD + m_tRP); }
  float getNsRowWrite() const { return m_tCK * (m_tWR + m_tRP); }
  float getNsTCCD() const { return m_tCK * m_tCCD_S; }
  float getNsAAP() const { return m_tCK * (m_tRAS + m_tRP); }

private:
  // [dram_structure]
  std::string m_protocol;
  int m_bankgroups = 0;
  int m_banksPerGroup = 0;
  int m_rows = 0;
  int m_columns = 0;
  int m_deviceWidth = 0;
  int m_BL = 0;

  // [timing]
  float m_tCK = 0.0;
  int m_AL = 0;
  int m_CL = 0;
  int m_CWL = 0;
  int m_tRCD = 0;
  int m_tRP = 0;
  int m_tRAS = 0;
  int m_tRFC = 0;
  int m_tRFC2 = 0;
  int m_tRFC4 = 0;
  int m_tREFI = 0;
  int m_tRPRE = 0;
  int m_tWPRE = 0;
  int m_tRRD_S = 0;
  int m_tRRD_L = 0;
  int m_tWTR_S = 0;
  int m_tWTR_L = 0;
  int m_tFAW = 0;
  int m_tWR = 0;
  int m_tWR2 = 0;
  int m_tRTP = 0;
  int m_tCCD_S = 0;
  int m_tCCD_L = 0;
  int m_tCKE = 0;
  int m_tCKESR = 0;
  int m_tXS = 0;
  int m_tXP = 0;
  int m_tRTRS = 0;

  // [power]
  float m_VDD = 0.0;
  int m_IDD0 = 0;
  float m_IPP0 = 0;
  int m_IDD2P = 0;
  int m_IDD2N = 0;
  int m_IDD3P = 0;
  int m_IDD3N = 0;
  int m_IDD4W = 0;
  int m_IDD4R = 0;
  int m_IDD5AB = 0;
  int m_IDD6x = 0;

  // [system]
  int m_channelSize = 0;
  int m_channels = 0;
  int m_busWidth = 0;
  std::string m_addressMapping;
  std::string m_queueStructure;
  std::string m_refreshPolicy;
  std::string m_rowBufPolicy;
  int m_cmdQueueSize = 0;
  int m_transQueueSize = 0;

  // [other]
  int m_epochPeriod = 0;
  int m_outputLevel = 0;
};

#endif

