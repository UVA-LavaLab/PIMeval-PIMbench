// File: pimPerfEnergyTables.h
// PIMeval Simulator - Performance and Energy Tables
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_PERF_ENERGY_TABLES_H
#define LAVA_PIM_PERF_ENERGY_TABLES_H

#include "libpimeval.h"
#include "pimCmd.h"
#include <unordered_map>
#include <tuple>


namespace pimPerfEnergyTables
{
  // Perf-energy table of BitSIMD-V variants
  extern const std::unordered_map<PimDeviceEnum, std::unordered_map<PimDataType,
      std::unordered_map<PimCmdEnum, std::tuple<unsigned, unsigned, unsigned>>>> bitsimdPerfTable;
}

#endif

