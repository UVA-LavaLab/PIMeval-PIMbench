// File: pimUtils.cc
// PIM Functional Simulator - Utilities
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimUtils.h"

//! @brief  Convert PimStatus enum to string
std::string
pimUtils::pimStatusEnumToStr(PimStatus status)
{
  switch (status) {
  case PIM_ERROR: return "ERROR";
  case PIM_OK: return "OK";
  }
  return "Unknown";
}

//! @brief  Convert PimDeviceEnum to string
std::string
pimUtils::pimDeviceEnumToStr(PimDeviceEnum deviceType)
{
  switch (deviceType) {
  case PIM_DEVICE_NONE: return "PIM_DEVICE_NONE";
  case PIM_FUNCTIONAL: return "PIM_FUNCTIONAL";
  case PIM_DEVICE_BITSIMD_V: return "PIM_DEVICE_BITSIMD_V";
  }
  return "Unknown";
}

//! @brief  Convert PimAllocEnum to string
std::string
pimUtils::pimAllocEnumToStr(PimAllocEnum allocType)
{
  switch (allocType) {
  case PIM_ALLOC_V1: return "PIM_ALLOC_V1";
  case PIM_ALLOC_H1: return "PIM_ALLOC_H1";
  }
  return "Unknown";
}

//! @brief  Convert PimCopyEnum to string
std::string
pimUtils::pimCopyEnumToStr(PimCopyEnum copyType)
{
  switch (copyType) {
  case PIM_COPY_V: return "PIM_COPY_V";
  case PIM_COPY_H: return "PIM_COPY_H";
  }
  return "Unknown";
}

//! @brief  Convert PimDataType enum to string
std::string
pimUtils::pimDataTypeEnumToStr(PimDataType dataType)
{
  switch (dataType) {
  case PIM_INT32: return "INT32";
  case PIM_INT64: return "INT64";
  }
  return "Unknown";
}

