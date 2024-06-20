// Bit-Serial Performance Modeling - BitSIMD_V_AP
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef BIT_SERIAL_BITSIMD_AP_H
#define BIT_SERIAL_BITSIMD_AP_H

#include "bitSerialBase.h"
#include "libpimsim.h"
#include <vector>

//! @class  bsBitsimd
//! @brief  Bit-serial perf for BitSIMD
class bitSerialBitsimdAp : public bitSerialBase
{
public:
  bitSerialBitsimdAp() {}
  ~bitSerialBitsimdAp() {}

protected:

  // virtual: create device
  virtual PimDeviceEnum getDeviceType() override { return PIM_DEVICE_BITSIMD_V_AP; }

private:

};

#endif
