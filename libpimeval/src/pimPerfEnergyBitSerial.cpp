// File: pimPerfEnergyBitSerial.cc
// PIMeval Simulator - Performance Energy Models
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimPerfEnergyBitSerial.h"
#include "pimCmd.h"
#include "pimPerfEnergyTables.h"
#include "pimUtils.h"
#include <iostream>
#include <cmath> // For log2()


//! @brief  Get performance and energy for bit-serial PIM
//!         BitSIMD and SIMDRAM need different fields
pimeval::perfEnergy
pimPerfEnergyBitSerial::getPerfEnergyBitSerial(PimDeviceEnum deviceType, PimCmdEnum cmdType, unsigned numPass, const pimObjInfo& objSrc1, const pimObjInfo& objSrc2, const pimObjInfo& objDest) const
{
  bool ok = false;
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numCores = objSrc1.getNumCoreAvailable();
  unsigned bitsPerElement = objSrc1.getBitsPerElement(PimBitWidth::ACTUAL);
  PimDataType dataType = objSrc1.getDataType();
  // workaround: special handling for pimAdd bool + bool = int
  if (cmdType == PimCmdEnum::ADD) {
    bitsPerElement = objDest.getBitsPerElement(PimBitWidth::ACTUAL);
    dataType = objDest.getDataType();
  }
  double msRead = 0.0;
  double msWrite = 0.0;
  double msLogic = 0.0;
  uint64_t totalOp = 0;

  switch (deviceType) {
    case PIM_DEVICE_BITSIMD_V:
    case PIM_DEVICE_BITSIMD_V_AP:
    case PIM_DEVICE_BITSIMD_H:
    {
      // BitSIMD-H reuse BitISMD-V perf for now
      if (deviceType == PIM_DEVICE_BITSIMD_H) {
        deviceType = PIM_DEVICE_BITSIMD_V;
      }
      // look up perf params from table
      unsigned numR = 0, numW = 0, numL = 0;
      auto it1 = pimPerfEnergyTables::bitsimdPerfTable.find(deviceType);
      if (it1 != pimPerfEnergyTables::bitsimdPerfTable.end()) {
        auto it2 = it1->second.find(dataType);
        if (it2 != it1->second.end()) {
          auto it3 = it2->second.find(cmdType);
          if (it3 != it2->second.end()) {
            numR = std::get<0>(it3->second);
            numW = std::get<1>(it3->second);
            numL = std::get<2>(it3->second);
            ok = true;
          }
        }
      }
      // workaround: adjust for add/sub mixed data type cases
      if (ok) {
        // pimAdd: int + bool = int, bool + bool = int
        // pimSub: int - bool = int
        if (cmdType == PimCmdEnum::ADD || cmdType == PimCmdEnum::SUB) {
          if (pimUtils::isSigned(dataType) || pimUtils::isUnsigned(dataType)) {
            unsigned numBitsSrc1 = pimUtils::getNumBitsOfDataType(objSrc1.getDataType(), PimBitWidth::ACTUAL);
            unsigned numBitsSrc2 = pimUtils::getNumBitsOfDataType(objSrc2.getDataType(), PimBitWidth::ACTUAL);
            numR = numBitsSrc1 + numBitsSrc2;
          }
        }
      }
      if (ok) {
        msRead += m_tR * numR;
        msWrite += m_tW * numW;
        msLogic += m_tL * numL;
        totalOp += (numL * objSrc1.getNumElements());
        msRuntime += msRead + msWrite + msLogic;
        mjEnergy += ((m_eL * numL * objSrc1.getMaxElementsPerRegion()) + (m_eAP * numR + m_eAP * numW)) * numCores;
        mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      }
      // handle bit-serial operations not in the above table
      if (!ok) {
        switch (cmdType) {
          case PimCmdEnum::COPY_O2O:
          {
            unsigned numR = bitsPerElement;
            unsigned numW = bitsPerElement;
            unsigned numL = 0;
            msRead += numR * m_tR;
            msWrite += numW * m_tW;
            msLogic += numL * m_tL;
            totalOp += (numL * objSrc1.getNumElements());
            msRuntime += msRead + msWrite + msLogic;
            mjEnergy += ((m_eL * numL * objSrc1.getMaxElementsPerRegion()) + (m_eAP * numR + m_eAP * numW)) * numCores;
            mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
            ok = true;
            break;
          }
          case PimCmdEnum::BIT_SLICE_EXTRACT:
          case PimCmdEnum::BIT_SLICE_INSERT:
            // each bit-slice extract/insert operation takes 1 row read and 1 row write
            msRead += m_tR;
            msWrite += m_tW;
            msRuntime += msRead + msWrite;
            mjEnergy += (m_eAP + m_eAP) * numCores;
            mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
            ok = true;
            break;
          case PimCmdEnum::COND_BROADCAST:
          {
            // bit-serial approach:
            // read the bool condition row, and move it to a bit register
            // for each row of dest:
            //   save the scalar value bit into a bit register
            //   read the row
            //   select between existing value and scalar value (more efficient with SEL instruction)
            //   write the row
            unsigned numR = 1 + bitsPerElement;
            unsigned numW = bitsPerElement;
            unsigned numL = 1 + 2 * bitsPerElement; // mov, (set, sel)
            msRead += numR * m_tR;
            msWrite += numW * m_tW;
            msLogic += numL * m_tL;
            totalOp += (numL * objSrc1.getNumElements());
            msRuntime += msRead + msWrite + msLogic;
            mjEnergy += ((m_eL * numL * objSrc1.getMaxElementsPerRegion()) + (m_eAP * numR + m_eAP * numW)) * numCores;
            mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
            ok = true;
            break;
          }
          case PimCmdEnum::SHIFT_BITS_L:
          case PimCmdEnum::SHIFT_BITS_R:
            // handle bit-shift specially
            msRead += m_tR * (bitsPerElement - 1);
            msWrite += m_tW * bitsPerElement;
            msLogic += m_tL;
            msRuntime += msRead + msWrite + msLogic;
            totalOp += objSrc1.getNumElements();
            mjEnergy += ((m_eL * objSrc1.getMaxElementsPerRegion()) + (m_eAP * numR + m_eAP * numW)) * numCores;
            mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
            ok = true;
            break;
          default:
            ; // pass
        }
      }
      break;
    }
    case PIM_DEVICE_SIMDRAM:
    {
      break;
    }
    default:
      assert(0);
  }

  if (!ok) {
    std::cout << "PIM-Warning: Unimplemented bit-serial runtime estimation for"
              << " device=" << pimUtils::pimDeviceEnumToStr(deviceType)
              << " cmd=" << pimCmd::getName(cmdType, "")
              << " dataType=" << pimUtils::pimDataTypeEnumToStr(dataType)
              << std::endl;
    msRuntime = 1000000;
  }
  msRead *= numPass;
  msWrite *= numPass;
  msLogic *= numPass;
  msRuntime *= numPass;
  mjEnergy *= numPass;
  totalOp *= numPass * numCores;
  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msLogic, totalOp);
}

//! @brief  Perf energy model of bit-serial type conversion
pimeval::perfEnergy
pimPerfEnergyBitSerial::getPerfEnergyTypeConversion(PimDeviceEnum deviceType, PimCmdEnum cmdType, const pimObjInfo& objSrc, const pimObjInfo& objDest) const
{
  assert(cmdType == PimCmdEnum::CONVERT_TYPE);
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  unsigned numCores = objSrc.getNumCoreAvailable();
  unsigned numPass = objSrc.getMaxNumRegionsPerCore();
  double msRead = 0.0;
  double msWrite = 0.0;
  double msLogic = 0.0;
  uint64_t totalOp = 0;

  switch (deviceType) {
    case PIM_DEVICE_BITSIMD_V:
    case PIM_DEVICE_BITSIMD_V_AP:
    case PIM_DEVICE_BITSIMD_H:
    {
      PimDataType dataTypeSrc = objSrc.getDataType();
      PimDataType dataTypeDest = objDest.getDataType();
      if (pimUtils::isFP(dataTypeSrc) || pimUtils::isFP(dataTypeDest)) {
        std::cout << "PIM-Warning: Unimplemented bit-serial runtime estimation for"
                  << " device=" << pimUtils::pimDeviceEnumToStr(m_simTarget)
                  << " cmd=" << pimCmd::getName(cmdType, "")
                  << " dataType=" << pimUtils::pimDataTypeEnumToStr(dataTypeSrc)
                  << std::endl;
        msRuntime = 1000000;
        break;
      }
      unsigned bitsPerElementSrc = objSrc.getBitsPerElement(PimBitWidth::ACTUAL);
      unsigned bitsPerElementDest = objDest.getBitsPerElement(PimBitWidth::ACTUAL);
      // integer type conversion
      unsigned numR = std::min(bitsPerElementSrc, bitsPerElementDest);
      unsigned numW = bitsPerElementDest;
      unsigned numL = 0;
      msRead = numR * m_tR;
      msWrite = numW * m_tW;
      msLogic = numL * m_tL;
      totalOp += (numL * objSrc.getNumElements());
      msRuntime = msRead + msWrite + msLogic;
      mjEnergy = ((m_eL * numL * objSrc.getMaxElementsPerRegion()) + (m_eAP * numR + m_eAP * numW)) * numCores;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    default:
      assert(0);
  }
  msRead *= numPass;
  msWrite *= numPass;
  msLogic *= numPass;
  msRuntime *= numPass;
  mjEnergy *= numPass;
  totalOp *= numPass * numCores;
  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msLogic, totalOp);
}

//! @brief  Perf energy model of bit-serial PIM for func1
pimeval::perfEnergy
pimPerfEnergyBitSerial::getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo& objSrc, const pimObjInfo& objDest) const
{
  pimeval::perfEnergy perf;
  switch (m_simTarget) {
    case PIM_DEVICE_BITSIMD_V:
    case PIM_DEVICE_BITSIMD_V_AP:
    case PIM_DEVICE_BITSIMD_H:
    case PIM_DEVICE_SIMDRAM:
    {
      // handle type conversion specially
      if (cmdType == PimCmdEnum::CONVERT_TYPE) {
        perf = getPerfEnergyTypeConversion(m_simTarget, cmdType, objSrc, objDest);
      } else {
        unsigned numPass = objSrc.getMaxNumRegionsPerCore();
        perf = getPerfEnergyBitSerial(m_simTarget, cmdType, numPass, objSrc, objSrc, objDest);
      }
      break;
    }
    default:
      assert(0);
  }
  return perf;
}

//! @brief  Perf energy model of bit-serial PIM for func2
pimeval::perfEnergy
pimPerfEnergyBitSerial::getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo& objSrc1, const pimObjInfo& objSrc2, const pimObjInfo& objDest) const
{
  pimeval::perfEnergy perf;
  switch (m_simTarget) {
    case PIM_DEVICE_BITSIMD_V:
    case PIM_DEVICE_BITSIMD_V_AP:
    case PIM_DEVICE_BITSIMD_H:
    case PIM_DEVICE_SIMDRAM:
    {
      unsigned numPass = objSrc1.getMaxNumRegionsPerCore();
      perf = getPerfEnergyBitSerial(m_simTarget, cmdType, numPass, objSrc1, objSrc2, objDest);
      break;
    }
    default:
      assert(0);
  }
  return perf;
}

//! @brief  Perf energy model of bit-serial PIM for reduction sum
pimeval::perfEnergy
pimPerfEnergyBitSerial::getPerfEnergyForReduction(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  uint64_t totalOp = 0;
  PimDataType dataType = obj.getDataType();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  uint64_t numElements = obj.getNumElements();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.getNumCoresUsed();
  double cpuTDP = 200; // W; AMD EPYC 9124 16 core

  switch (m_simTarget) {
    case PIM_DEVICE_BITSIMD_V:
    case PIM_DEVICE_BITSIMD_V_AP:
    {
      if (pimUtils::isSigned(dataType) || pimUtils::isUnsigned(dataType)) {
        switch (cmdType)
        {
        case PimCmdEnum::REDSUM:
        case PimCmdEnum::REDSUM_RANGE:
        {
          // Assume row-wide popcount capability for integer reduction, with a 64-bit popcount logic unit per PIM core
          // For a single row, popcount is calculated per 64-bit chunks, and result is shifted then added to an 64-bit accumulator register
          // If there are multiple regions per core, the multi-region reduction sum is stored in the accumulator
          // reduction for all regions
          double aggregateMs = static_cast<double>(numCore) / 3200000;
          double mjEnergyPerPcl = m_pclNsDelay * m_pclUwPower * 1e-12;
          int numPclPerCore = (maxElementsPerRegion + 63) / 64; // number of 64-bit popcount needed for a row
          msRead = m_tR * bitsPerElement * numPass;
          msWrite = 0;
          msCompute = aggregateMs + ((m_pclNsDelay * 1e-6) * numPclPerCore * bitsPerElement * numPass) ; 
          mjEnergy = m_eAP * numCore + mjEnergyPerPcl * numPclPerCore * numCore; // energy of one row read and row-wide popcount
          mjEnergy *= bitsPerElement * numPass;
          msRuntime += msRead + msWrite + msCompute;
          mjEnergy += aggregateMs * cpuTDP;
          mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
          totalOp += (numPclPerCore * bitsPerElement * numPass * numCore);
          break;
        }
        case PimCmdEnum::REDMIN:
        case PimCmdEnum::REDMIN_RANGE:
        {
          // Reduction tree approach.
          // `numpass` for one reduction min is halved because of the reduction tree based approach.
          // The following does not consider the cost for data rearrangement. Ideally that should be considered.
          // TODO: for ranged reduction, `numElements` should be the #elements in the range
          unsigned levels = static_cast<unsigned>(std::ceil(std::log2(numElements))); // Tree depth
          pimeval::perfEnergy perfEnergyBS = getPerfEnergyBitSerial(m_simTarget, cmdType, (std::ceil(numPass*1.0/2)), obj, obj, obj);
          msRuntime = perfEnergyBS.m_msRuntime * levels;
          mjEnergy = perfEnergyBS.m_mjEnergy * levels;
          msRead = perfEnergyBS.m_msRead * levels;
          msWrite = perfEnergyBS.m_msWrite * levels;
          msCompute = perfEnergyBS.m_msCompute * levels;
          totalOp += perfEnergyBS.m_totalOp * levels;
          break;
        }
        case PimCmdEnum::REDMAX:
        case PimCmdEnum::REDMAX_RANGE:
        {
          // Reduction tree approach.
          // `numpass` for one reduction min is halved because of the reduction tree based approach.
          // The following does not consider the cost for data rearrangement. Ideally that should be considered.
          // TODO: for ranged reduction, `numElements` should be the #elements in the range
          unsigned levels = static_cast<unsigned>(std::ceil(std::log2(numElements))); // Tree depth
          pimeval::perfEnergy perfEnergyBS = getPerfEnergyBitSerial(m_simTarget, cmdType, (std::ceil(numPass*1.0/2)), obj, obj, obj);
          msRuntime = perfEnergyBS.m_msRuntime * levels;
          mjEnergy = perfEnergyBS.m_mjEnergy * levels;
          msRead = perfEnergyBS.m_msRead * levels;
          msWrite = perfEnergyBS.m_msWrite * levels;
          msCompute = perfEnergyBS.m_msCompute * levels;
          totalOp += perfEnergyBS.m_totalOp * levels;
          break;
        }
        default:
        {
          std::cout << "PIM-Warning: Unsupported reduction command for bit-serial PIM: "
                    << pimCmd::getName(cmdType, "") << std::endl;
          break;
        }
        }
      } else if (pimUtils::isFP(dataType)) {
        std::cout << "PIM-Warning: Perf energy model for FP reduction sum on bit-serial PIM is not available yet." << std::endl;
        msRuntime = 999999999.9; // todo
        mjEnergy = 999999999.9;  // todo
      } else {
        assert(0);
      }
      break;
    }
    case PIM_DEVICE_SIMDRAM:
      // todo
      std::cout << "PIM-Warning: SIMDRAM performance stats not implemented yet." << std::endl;
      break;
    case PIM_DEVICE_BITSIMD_H:
      // Sequentially process all elements per CPU cycle
      msRuntime = static_cast<double>(numElements) / 3200000; // typical 3.2 GHz CPU
      mjEnergy = 999999999.9; // todo
      // consider PCL
      break;
    default:
      assert(0);
  }
  
  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}

//! @brief  Perf energy model of bit-serial PIM for broadcast
pimeval::perfEnergy
pimPerfEnergyBitSerial::getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.getNumCoresUsed();
  uint64_t totalOp = 0;
  switch (m_simTarget) {
    case PIM_DEVICE_BITSIMD_V:
    case PIM_DEVICE_BITSIMD_V_AP:
    {
      // For one pass: For every bit: Set SA to bit value; Write SA to row;
      msRead = 0;
      msWrite = m_tW * bitsPerElement * numPass;
      msCompute = m_tL * bitsPerElement * numPass;
      totalOp = bitsPerElement * numPass * numCore * obj.getNumElements();
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = m_eAP * numCore * numPass ;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    case PIM_DEVICE_SIMDRAM:
    {
      // todo
      msRuntime *= numPass;
      mjEnergy *= numPass;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    case PIM_DEVICE_BITSIMD_H:
    {
      // For one pass: For every element: 1 tCCD per byte
      uint64_t maxBytesPerRegion = (uint64_t)maxElementsPerRegion * (bitsPerElement / 8);
      msRuntime = m_tW + m_tL * maxBytesPerRegion; // for one pass
      msRuntime *= numPass;
      mjEnergy = (m_eAP + (m_tL * maxBytesPerRegion)) * numCore * numPass;
      mjEnergy += m_pBChip * m_numChipsPerRank * m_numRanks * msRuntime;
      break;
    }
    default:
      assert(0);
  }
  
  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}

//! @brief  Perf energy model of bit-serial PIM for rotate
pimeval::perfEnergy
pimPerfEnergyBitSerial::getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  double mjEnergy = 0.0;
  double msRead = 0.0;
  double msWrite = 0.0;
  double msCompute = 0.0;
  uint64_t totalOp = 0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement(PimBitWidth::ACTUAL);
  unsigned numRegions = obj.getRegions().size();
  unsigned numCore = obj.getNumCoreAvailable();
  // boundary handling - assume two times copying between device and host for boundary elements
  pimeval::perfEnergy perfEnergyBT = getPerfEnergyForBytesTransfer(PimCmdEnum::COPY_D2H, numRegions * bitsPerElement / 8);

  switch (m_simTarget) {
    case PIM_DEVICE_BITSIMD_V:
    case PIM_DEVICE_BITSIMD_V_AP:
      // rotate within subarray:
      // For every bit: Read row to SA; move SA to R1; Shift R1; Move R1 to SA; Write SA to row
      msRead = m_tR * bitsPerElement * numPass;
      msWrite = m_tW * bitsPerElement * numPass;
      msCompute = 3 * m_tL * bitsPerElement * numPass;
      totalOp += 3 * bitsPerElement * numPass * numCore;
      msRuntime = msRead + msWrite + msCompute;
      mjEnergy = (m_eAP + 3 * m_eL) * bitsPerElement * numPass; // for one pass
      msRuntime += 2 * perfEnergyBT.m_msRuntime;
      mjEnergy += 2 * perfEnergyBT.m_mjEnergy;
      break;
    case PIM_DEVICE_SIMDRAM:
      // todo
      break;
    case PIM_DEVICE_BITSIMD_H:
      // rotate within subarray:
      // For every bit: Read row to SA; move SA to R1; Shift R1 by N steps; Move R1 to SA; Write SA to row
      // TODO: separate bank level and GDL
      // TODO: energy unimplemented
      // TODO: R,W,L calcutation
      // TOD): total Op uinimplemented
      msRuntime = (m_tR + (bitsPerElement + 2) * m_tL + m_tW); // for one pass
      msRuntime *= numPass;
      mjEnergy = (m_eAP + (bitsPerElement + 2) * m_eL) * numPass;
      msRuntime += 2 * perfEnergyBT.m_msRuntime;
      mjEnergy += 2 * perfEnergyBT.m_mjEnergy;
      break;
    default:
      assert(0);
  }
  
  return pimeval::perfEnergy(msRuntime, mjEnergy, msRead, msWrite, msCompute, totalOp);
}

