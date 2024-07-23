// Bit-Serial Performance Modeling - SIMDRAM
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef BIT_SERIAL_SIMDRAM_H
#define BIT_SERIAL_SIMDRAM_H

#include "bitSerialBase.h"
#include "libpimeval.h"
#include <vector>

//! @class  bitSerialSimdram
//! @brief  Bit-serial perf for SIMDRAM
class bitSerialSimdram : public bitSerialBase
{
public:
  bitSerialSimdram() { m_deviceName = "simdram"; }
  ~bitSerialSimdram() {}

protected:

  // virtual: create device
  virtual PimDeviceEnum getDeviceType() override { return PIM_DEVICE_SIMDRAM; }

private:

};

#endif
