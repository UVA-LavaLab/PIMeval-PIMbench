// File: pimParamsDram.h
// PIMeval Simulator - DRAM parameters
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_PARAMS_DRAM_H
#define LAVA_PIM_PARAMS_DRAM_H

#include <string>
#include <memory>
#include "libpimeval.h"

//! @class  pimParamsDram
//! @brief  DRAM parameters (DRAMsim3 compatible)
class pimParamsDram
{
public:
  // Virtual destructor to ensure derived class destructors are called
  virtual ~pimParamsDram() = default;

  // Static factory method to create appropriate subclass based on protocol
  static std::unique_ptr<pimParamsDram> create(PimDeviceProtocolEnum deviceProtocol);

  // Static factory method to create appropriate subclass based on config file
  static std::unique_ptr<pimParamsDram> createFromConfig(const std::string& config);

  // Virtual functions for protocol-specific implementation
  virtual int getDeviceWidth() const = 0;
  virtual int getBurstLength() const = 0;
  virtual int getNumChipsPerRank() const = 0;
  virtual double getNsRowRead() const = 0;
  virtual double getNsRowWrite() const = 0;
  virtual double getNsTCCD_S() const = 0;
  virtual double getNsTCAS() const = 0;
  virtual double getNsAAP() const = 0;
  virtual double getTypicalRankBW() const = 0;
  virtual double getPjRowRead() const = 0;
  virtual double getPjLogic() const = 0;
  virtual double getMwIDD2N() const = 0;
  virtual double getMwIDD3N() const = 0;
  virtual double getMwRead() const = 0;
  virtual double getMwWrite() const = 0;
};

#endif

