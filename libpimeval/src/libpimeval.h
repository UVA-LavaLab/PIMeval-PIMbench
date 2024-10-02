// File: libpimeval.h
// PIMeval Simulator - Library Interface
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_LIB_PIM_EVAL_H
#define LAVA_LIB_PIM_EVAL_H

#include <cstdint>
#include <cstdarg>

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
  PIM_DEVICE_BITSIMD_V_NAND,
  PIM_DEVICE_BITSIMD_V_MAJ,
  PIM_DEVICE_BITSIMD_V_AP,
  PIM_DEVICE_DRISA_NOR,
  PIM_DEVICE_DRISA_MIXED,
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
  PIM_BOOL = 0,
  PIM_INT8,
  PIM_INT16,
  PIM_INT32,
  PIM_INT64,
  PIM_UINT8,
  PIM_UINT16,
  PIM_UINT32,
  PIM_UINT64,
  PIM_FP32,
};

//! @brief  PIM device properties
struct PimDeviceProperties {
  PimDeviceEnum deviceType = PIM_DEVICE_NONE;
  unsigned numRanks = 0;
  unsigned numBankPerRank = 0;
  unsigned numSubarrayPerBank = 0;
  unsigned numRowPerSubarray = 0;
  unsigned numColPerSubarray = 0;
};

typedef int PimCoreId;
typedef int PimObjId;

// Device creation and deletion
PimStatus pimCreateDevice(PimDeviceEnum deviceType, unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank, unsigned numRows, unsigned numCols);
PimStatus pimCreateDeviceFromConfig(PimDeviceEnum deviceType, const char* configFileName);
PimStatus pimGetDeviceProperties(PimDeviceProperties* deviceProperties);
PimStatus pimDeleteDevice();
void pimShowStats();
void pimResetStats();

// Resource allocation and deletion
PimObjId pimAlloc(PimAllocEnum allocType, uint64_t numElements, PimDataType dataType);
PimObjId pimAllocAssociated(PimObjId assocId, PimDataType dataType);
PimStatus pimFree(PimObjId obj);
PimObjId pimCreateRangedRef(PimObjId refId, uint64_t idxBegin, uint64_t idxEnd);

// Data transfer
// Note: idxBegin and idxEnd specify the range of indexes to be processed by the PIM.
// The size of the host-side vector should match the size of this range on the PIM side.
// If the default values for idxBegin and idxEnd are used, the entire range of the PIM object will be considered.
PimStatus pimCopyHostToDevice(void* src, PimObjId dest, uint64_t idxBegin = 0, uint64_t idxEnd = 0);
PimStatus pimCopyDeviceToHost(PimObjId src, void* dest, uint64_t idxBegin = 0, uint64_t idxEnd = 0);
PimStatus pimCopyHostToDeviceWithType(PimCopyEnum copyType, void* src, PimObjId dest, uint64_t idxBegin = 0, uint64_t idxEnd = 0);
PimStatus pimCopyDeviceToHostWithType(PimCopyEnum copyType, PimObjId src, void* dest, uint64_t idxBegin = 0, uint64_t idxEnd = 0);
PimStatus pimCopyDeviceToDevice(PimObjId src, PimObjId dest, uint64_t idxBegin = 0, uint64_t idxEnd = 0);

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
PimStatus pimAddScalar(PimObjId src, PimObjId dest, uint64_t scalerValue);
PimStatus pimSubScalar(PimObjId src, PimObjId dest, uint64_t scalerValue);
PimStatus pimMulScalar(PimObjId src, PimObjId dest, uint64_t scalerValue);
PimStatus pimDivScalar(PimObjId src, PimObjId dest, uint64_t scalerValue);
PimStatus pimAndScalar(PimObjId src, PimObjId dest, uint64_t scalerValue);
PimStatus pimOrScalar(PimObjId src, PimObjId dest, uint64_t scalerValue);
PimStatus pimXorScalar(PimObjId src, PimObjId dest, uint64_t scalerValue);
PimStatus pimXnorScalar(PimObjId src, PimObjId dest, uint64_t scalerValue);
PimStatus pimGTScalar(PimObjId src, PimObjId dest, uint64_t scalerValue);
PimStatus pimLTScalar(PimObjId src, PimObjId dest, uint64_t scalerValue);
PimStatus pimEQScalar(PimObjId src, PimObjId dest, uint64_t scalerValue);
PimStatus pimMinScalar(PimObjId src, PimObjId dest, uint64_t scalerValue);
PimStatus pimMaxScalar(PimObjId src, PimObjId dest, uint64_t scalerValue);
// multiply src1 with scalerValue and add the multiplication result with src2. Save the result to dest. 
PimStatus pimScaledAdd(PimObjId src1, PimObjId src2, PimObjId dest, uint64_t scalerValue);
PimStatus pimPopCount(PimObjId src, PimObjId dest);
PimStatus pimRedSumInt(PimObjId src, int64_t* sum);
PimStatus pimRedSumUInt(PimObjId src, uint64_t* sum);
// Note: Reduction sum range is [idxBegin, idxEnd)
PimStatus pimRedSumRangedInt(PimObjId src, uint64_t idxBegin, uint64_t idxEnd, int64_t* sum);
PimStatus pimRedSumRangedUInt(PimObjId src, uint64_t idxBegin, uint64_t idxEnd, uint64_t* sum);
PimStatus pimBroadcastInt(PimObjId dest, int64_t value);
PimStatus pimBroadcastUInt(PimObjId dest, uint64_t value);
PimStatus pimRotateElementsRight(PimObjId src);
PimStatus pimRotateElementsLeft(PimObjId src);
PimStatus pimShiftElementsRight(PimObjId src);
PimStatus pimShiftElementsLeft(PimObjId src);
PimStatus pimShiftBitsRight(PimObjId src, PimObjId dest, unsigned shiftAmount);
PimStatus pimShiftBitsLeft(PimObjId src, PimObjId dest, unsigned shiftAmount);

////////////////////////////////////////////////////////////////////////////////
// Warning: Do not use below micro-ops level APIs for functional simulation   //
////////////////////////////////////////////////////////////////////////////////

// BitSIMD micro ops
// Note: Below APIs are for low-level micro-ops programming but not for functional simulation
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

// Customized allocation with bitsPerElement less than the number of bits of dataType
PimObjId pimAllocCustomized(PimAllocEnum allocType, uint64_t numElements, PimDataType dataType, unsigned bitsPerElement);
PimObjId pimAllocAssociatedCustomized(PimObjId assocId, PimDataType dataType, unsigned bitsPerElement);
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
//   - Example: pimOpAAP(1, 2, DCC0N, 0, T0, 0, T3, 0) // T0, T3 = DCC0N
// Requirements:
//   - numSrc must be odd (1 or 3) to perform MAJ operation
//   - Number of var args must be 2*numSrc for AP and 2*(numDest+numSrc) for AAP
//   - Var args must be a list of (PimObjId, unsigned ofst) pairs
// Dual contact ref:
//   - Warning: Dual contact ref is only supported by pimCopy/AP/AAP for now
PimObjId pimCreateDualContactRef(PimObjId refId);
PimStatus pimOpAP(int numSrc, ...);
PimStatus pimOpAAP(int numSrc, int numDest, ...);

#endif

