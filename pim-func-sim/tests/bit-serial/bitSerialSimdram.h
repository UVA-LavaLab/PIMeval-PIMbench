// Bit-Serial Performance Modeling - SIMDRAM
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef BIT_SERIAL_SIMDRAM_H
#define BIT_SERIAL_SIMDRAM_H

#include "bitSerialBase.h"
#include "libpimsim.h"
#include <vector>

//! @class  bitSerialSimdram
//! @brief  Bit-serial perf for SIMDRAM
class bitSerialSimdram : public bitSerialBase
{
public:
  bitSerialSimdram() {}
  ~bitSerialSimdram() {}

protected:

  // virtual: create device
  virtual PimDeviceEnum getDeviceType() override { return PIM_DEVICE_NONE; }

private:

};

#endif
