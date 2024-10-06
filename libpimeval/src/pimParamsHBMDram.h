// File: pimParamsHBMDram.h
// PIMeval Simulator - HBM DRAM parameters
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_PARAMS_HBM_DRAM_H
#define LAVA_PIM_PARAMS_HBM_DRAM_H

#include <string>
#include <unordered_map>
#include "pimParamsDram.h"

//! @class  pimParamsDDRDram
//! @brief  DRAM parameters (DRAMsim3 compatible)
class pimParamsHBMDram : public pimParamsDram
{
public:
  pimParamsHBMDram();
  pimParamsHBMDram(std::unordered_map<std::string, std::string> params);
  ~pimParamsHBMDram() override = default;

  int getDeviceWidth() const override { return m_deviceWidth;}
  int getBurstLength() const override { return m_BL;}
  int getNumChipsPerRank() const override {return m_busWidth / m_deviceWidth; }
  float getNsRowRead() const override { return m_tCK * (m_tRCDRD + m_tRP); }
  float getNsRowWrite() const override { return m_tCK * (m_tWR + m_tRP + m_tRCDWR); }
  float getNsTCCD_S() const override { return m_tCK * m_tCCD_S; }
  float getNsTCAS() const override { return m_tCK * m_CL; }
  float getNsAAP() const override { return m_tCK * (m_tRAS + m_tRP); }
  double getTypicalRankBW() const override { return m_typicalRankBW; }
  double getPjRowRead() const override { return m_VDD * (m_IDD0 * (m_tRAS + m_tRP) - (m_IDD3N * m_tRAS + m_IDD2N * m_tRP)); } // Energy for 1 Activate command (and the correspound precharge command) in one subarray of one bank of one chip
  double getPjLogic() const override { return 0.007 * m_tCK * m_tCCD_S ; } // 0.007 mW is the total power per BSLU, 0.007 * m_tCK * m_tCCD_S is the energy of one BSLU during one logic operation in pJ.
  double getMwIDD2N() const override {return m_VDD * m_IDD2N; }
  double getMwIDD3N() const override {return m_VDD * m_IDD3N; }
  double getMwRead() const override { return m_VDD * (m_IDD4R - m_IDD3N); } // read power per chip (data copy)
  double getMwWrite() const override { return m_VDD * (m_IDD4W - m_IDD3N); } // write power per chip (data copy)

private:
  // [dram_structure]
  std::string m_protocol;
  int m_bankgroups = 0;
  int m_banksPerGroup = 0;
  int m_rows = 0;
  int m_columns = 0;
  int m_deviceWidth = 0;
  int m_BL = 0;
  int m_numDies = 0;

  // [timing]
  float m_tCK = 0.0;
  int m_CL = 0;
  int m_CWL = 0;
  int m_tRCDRD = 0;
  int m_tRCDWR = 0;
  int m_tRP = 0;
  int m_tRAS = 0;
  int m_tRFC = 0;
  int m_tREFI = 0;
  int m_tREFIb = 0;
  int m_tRPRE = 0;
  int m_tWPRE = 0;
  int m_tRRD_S = 0;
  int m_tRRD_L = 0;
  int m_tWTR_S = 0;
  int m_tWTR_L = 0;
  int m_tFAW = 0;
  int m_tWR = 0;
  int m_tCCD_S = 0;
  int m_tCCD_L = 0;
  int m_tXS = 0;
  int m_tCKE = 0;
  int m_tCKESR = 0;
  int m_tXP = 0;
  int m_tRTP_L = 0;
  int m_tRTP_S = 0;

  // [power]
  float m_VDD = 0.0;
  int m_IDD0 = 0;
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
  std::string m_rowBufPolicy;
  int m_cmdQueueSize = 0;
  int m_transQueueSize = 0;
  bool m_unifiedQueue = false;

  // [other]
  int m_epochPeriod = 0;
  int m_outputLevel = 0;

  // Extended
  double m_typicalRankBW = 25.6; // GB/s
};

#endif

