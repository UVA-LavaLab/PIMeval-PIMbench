// File: libpimsim.h
// PIM Functional Simulator Library Interface
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_LIB_PIM_SIM_H
#define LAVA_LIB_PIM_SIM_H

#ifdef __cplusplus
#include <cstdarg>
#else
#include <stdarg.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

  //! @brief  PIM API return status
  enum PimStatus {
    PIM_ERROR = 0,
    PIM_OK,
  };

  //! @brief  PIM device types
  enum PimDeviceEnum {
    PIM_DEVICE_NONE = 0,
    PIM_FUNCTIONAL,
    PIM_DEVICE_BITSIMD_V,
    PIM_DEVICE_BITSIMD_V_AP,
    PIM_DEVICE_SIMDRAM,
    PIM_DEVICE_BITSIMD_H,
    PIM_DEVICE_FULCRUM,
    PIM_DEVICE_BANK_LEVEL,
  };

  //! @brief  PIM allocation types
  enum PimAllocEnum {
    PIM_ALLOC_AUTO = 0, // Auto determine vertical or horizontal layout based on device type
    PIM_ALLOC_V,        // V layout, multiple regions per core
    PIM_ALLOC_H,        // H layout, multiple regions per core 
    PIM_ALLOC_V1,       // V layout, at most 1 region per core
    PIM_ALLOC_H1,       // H layout, at most 1 region per core
  };

  //! @brief  PIM data copy types
  enum PimCopyEnum {
    PIM_COPY_V,
    PIM_COPY_H,
  };

  //! @brief  PIM datatypes
  enum PimDataType {
    PIM_INT32 = 0,
    PIM_INT64,
    PIM_FP32,
  };

  typedef int PimCoreId;
  typedef int PimObjId;

  // Device creation and deletion
  PimStatus pimCreateDevice(PimDeviceEnum deviceType, unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank, unsigned numRows, unsigned numCols);
  PimStatus pimCreateDeviceFromConfig(PimDeviceEnum deviceType, const char* configFileName);
  PimStatus pimDeleteDevice();
  void pimShowStats();

  // Resource allocation and deletion
  PimObjId pimAlloc(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement, PimDataType dataType);
  PimObjId pimAllocAssociated(unsigned bitsPerElement, PimObjId assocId, PimDataType dataType);
  PimStatus pimFree(PimObjId obj);
  PimObjId pimCreateRangedRef(PimObjId refId, unsigned idxBegin, unsigned idxEnd);
  PimObjId pimCreateDualContactRef(PimObjId refId);

  // Data transfer
  PimStatus pimCopyHostToDevice(void* src, PimObjId dest);
  PimStatus pimCopyDeviceToHost(PimObjId src, void* dest);
  PimStatus pimCopyHostToDeviceWithType(PimCopyEnum copyType, void* src, PimObjId dest);
  PimStatus pimCopyDeviceToHostWithType(PimCopyEnum copyType, PimObjId src, void* dest);
  PimStatus pimCopyDeviceToDevice(PimObjId src, PimObjId dest);

  // Logic and Arithmetic Operation
  PimStatus pimAdd(PimObjId src1, PimObjId src2, PimObjId dest);
  PimStatus pimSub(PimObjId src1, PimObjId src2, PimObjId dest);
  PimStatus pimMul(PimObjId src1, PimObjId src2, PimObjId dest);
  PimStatus pimDiv(PimObjId src1, PimObjId src2, PimObjId dest);
  PimStatus pimAbs(PimObjId src, PimObjId dest);
  PimStatus pimAnd(PimObjId src1, PimObjId src2, PimObjId dest);
  PimStatus pimOr(PimObjId src1, PimObjId src2, PimObjId dest);
  PimStatus pimXor(PimObjId src1, PimObjId src2, PimObjId dest);
  PimStatus pimXnor(PimObjId src1, PimObjId src2, PimObjId dest);
  PimStatus pimGT(PimObjId src1, PimObjId src2, PimObjId dest);
  PimStatus pimLT(PimObjId src1, PimObjId src2, PimObjId dest);
  PimStatus pimEQ(PimObjId src1, PimObjId src2, PimObjId dest);
  PimStatus pimMin(PimObjId src1, PimObjId src2, PimObjId dest);
  PimStatus pimMax(PimObjId src1, PimObjId src2, PimObjId dest);
  PimStatus pimPopCount(PimObjId src, PimObjId dest);
  PimStatus pimRedSum(PimObjId src, int* sum);
  PimStatus pimRedSumRanged(PimObjId src, unsigned idxBegin, unsigned idxEnd, int* sum);
  PimStatus pimBroadcast(PimObjId dest, unsigned value);
  PimStatus pimRotateElementsRight(PimObjId src);
  PimStatus pimRotateElementsLeft(PimObjId src);
  PimStatus pimShiftElementsRight(PimObjId src);
  PimStatus pimShiftElementsLeft(PimObjId src);
  PimStatus pimShiftBitsRight(PimObjId src, PimObjId dest, unsigned shiftAmount);
  PimStatus pimShiftBitsLeft(PimObjId src, PimObjId dest, unsigned shiftAmount);

  // BitSIMD-V: Row-wide bit registers per subarray
  enum PimRowReg {
    PIM_RREG_NONE = 0,
    PIM_RREG_SA,
    PIM_RREG_R1,
    PIM_RREG_R2,
    PIM_RREG_R3,
    PIM_RREG_R4,
    PIM_RREG_R5,
  };

  // BitSIMD-V micro ops
  PimStatus pimOpReadRowToSa(PimObjId src, unsigned ofst);
  PimStatus pimOpWriteSaToRow(PimObjId src, unsigned ofst);
  PimStatus pimOpTRA(PimObjId src1, unsigned ofst1, PimObjId src2, unsigned ofst2, PimObjId src3, unsigned ofst3);
  PimStatus pimOpMove(PimObjId objId, PimRowReg src, PimRowReg dest);
  PimStatus pimOpSet(PimObjId objId, PimRowReg src, bool val);
  PimStatus pimOpNot(PimObjId objId, PimRowReg src, PimRowReg dest);
  PimStatus pimOpAnd(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  PimStatus pimOpOr(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  PimStatus pimOpNand(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  PimStatus pimOpNor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  PimStatus pimOpXor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  PimStatus pimOpXnor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  PimStatus pimOpMaj(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg src3, PimRowReg dest);
  PimStatus pimOpSel(PimObjId objId, PimRowReg cond, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  PimStatus pimOpRotateRH(PimObjId objId, PimRowReg src);
  PimStatus pimOpRotateLH(PimObjId objId, PimRowReg src);

  // SIMDRAM micro ops
  // AP:
  //   - Functionality: {srcRows} = MAJ(srcRows)
  //   - Action: Activate srcRows simultaneously, followed by a precharge
  //   - Example: pimOpAP(3, T0, 0, T1, 0, T2, 0) // T0, T1, T2 = MAJ(T0, T1, T2)
  // AAP:
  //   - Functionality: {srcRows, destRows} = MAJ(srcRows)
  //   - Action: Activate srcRows simultaneously, copy result to all destRows, followed by a precharge 
  //   - Example: pimOpAAP(2, 1, T0, 0, T3, 0, DCC0N, 0) // T0, T3 = DCC0N
  // Requirements:
  //   - numSrc must be odd (1 or 3) to perform MAJ operation
  //   - Number of var args must be 2*numSrc for AP and 2*(numDest+numSrc) for AAP
  //   - Var args must be a list of (PimObjId, int ofst) pairs
  PimStatus pimOpAP(int numSrc, ...);
  PimStatus pimOpAAP(int numDest, int numSrc, ...);
  
#ifdef __cplusplus
}
#endif

#endif

