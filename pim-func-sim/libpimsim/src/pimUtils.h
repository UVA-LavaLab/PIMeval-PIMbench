// File: pimUtils.h
// PIM Functional Simulator - Utilities
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_UTILS_H
#define LAVA_PIM_UTILS_H

#include "libpimsim.h"
#include <string>

namespace pimUtils
{
  std::string pimStatusEnumToStr(PimStatus status);
  std::string pimDeviceEnumToStr(PimDeviceEnum deviceType);
  std::string pimAllocEnumToStr(PimAllocEnum allocType);
  std::string pimCopyEnumToStr(PimCopyEnum copyType);
  std::string pimDataTypeEnumToStr(PimDataType dataType);
}

#endif

