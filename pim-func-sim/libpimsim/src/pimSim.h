// File: pimSim.h
// PIM Functional Simulator - PIM Simulator Main Entry
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_SIM_H
#define LAVA_PIM_SIM_H

#include "libpimsim.h"
#include "pimDevice.h"
#include "pimParamsDram.h"
#include "pimParamsPerf.h"
#include "pimStats.h"
#include <vector>
#include <cstdarg>


//! @class  pimSim
//! @brief  PIM simulator singleton class
class pimSim
{
public:
  static pimSim* get();
  static void destroy();

  // Device creation and deletion
  bool createDevice(PimDeviceEnum deviceType, unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank, unsigned numRows, unsigned numCols);
  bool createDeviceFromConfig(PimDeviceEnum deviceType, const char* configFileName);
  bool deleteDevice();
  bool isValidDevice(bool showMsg = true) const;

  PimDeviceEnum getDeviceType() const;
  PimDeviceEnum getSimTarget() const;
  unsigned getNumRanks() const;
  unsigned getNumBankPerRank() const;
  unsigned getNumSubarrayPerBank() const;
  unsigned getNumRowPerSubarray() const;
  unsigned getNumColPerSubarray() const;
  unsigned getNumCores() const;
  unsigned getNumRows() const;
  unsigned getNumCols() const;

  void showStats() const;
  void resetStats() const;
  pimStatsMgr* getStatsMgr() { return m_statsMgr; }
  pimParamsDram* getParamsDram() { return m_paramsDram; }
  pimParamsPerf* getParamsPerf() { return m_paramsPerf; }
  pimUtils::threadPool* getThreadPool() { return m_threadPool; }
  unsigned getNumThreads() const { return m_numThreads; }

  // Resource allocation and deletion
  PimObjId pimAlloc(PimAllocEnum allocType, uint64_t numElements, unsigned bitsPerElement, PimDataType dataType);
  PimObjId pimAllocAssociated(unsigned bitsPerElement, PimObjId assocId, PimDataType dataType);
  bool pimFree(PimObjId obj);
  PimObjId pimCreateRangedRef(PimObjId refId, uint64_t idxBegin, uint64_t idxEnd);
  PimObjId pimCreateDualContactRef(PimObjId refId);

  // Data transfer
  bool pimCopyMainToDevice(void* src, PimObjId dest);
  bool pimCopyDeviceToMain(PimObjId src, void* dest);
  bool pimCopyMainToDeviceWithType(PimCopyEnum copyType, void* src, PimObjId dest);
  bool pimCopyDeviceToMainWithType(PimCopyEnum copyType, PimObjId src, void* dest);
  bool pimCopyDeviceToDevice(PimObjId src, PimObjId dest);

  // Computation
  bool pimAdd(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimSub(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimDiv(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimAbs(PimObjId src, PimObjId dest);
  bool pimMul(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimOr(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimAnd(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimXor(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimXnor(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimGT(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimLT(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimEQ(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimMin(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimMax(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimAdd(PimObjId src, PimObjId dest, int64_t scalerValue);
  bool pimSub(PimObjId src, PimObjId dest, int64_t scalerValue);
  bool pimMul(PimObjId src, PimObjId dest, int64_t scalerValue);
  bool pimDiv(PimObjId src, PimObjId dest, int64_t scalerValue);
  bool pimAnd(PimObjId src, PimObjId dest, int64_t scalerValue);
  bool pimOr(PimObjId src, PimObjId dest, int64_t scalerValue);
  bool pimXor(PimObjId src, PimObjId dest, int64_t scalerValue);
  bool pimXnor(PimObjId src, PimObjId dest, int64_t scalerValue);
  bool pimGT(PimObjId src, PimObjId dest, int64_t scalerValue);
  bool pimLT(PimObjId src, PimObjId dest, int64_t scalerValue);
  bool pimEQ(PimObjId src, PimObjId dest, int64_t scalerValue);
  bool pimMin(PimObjId src, PimObjId dest, int64_t scalerValue);
  bool pimMax(PimObjId src, PimObjId dest, int64_t scalerValue);
  bool pimPopCount(PimObjId src, PimObjId dest);
  template <typename T> bool pimRedSum(PimObjId src, T* sum);
  template <typename T> bool pimRedSumRanged(PimObjId src, uint64_t idxBegin, uint64_t idxEnd, T* sum);
  template <typename T> bool pimBroadcast(PimObjId dest, T value);
  bool pimRotateElementsRight(PimObjId src);
  bool pimRotateElementsLeft(PimObjId src);
  bool pimShiftElementsRight(PimObjId src);
  bool pimShiftElementsLeft(PimObjId src);
  bool pimShiftBitsRight(PimObjId src, PimObjId dest, unsigned shiftAmount);
  bool pimShiftBitsLeft(PimObjId src, PimObjId dest, unsigned shiftAmount);

  // BitSIMD-V micro ops
  bool pimOpReadRowToSa(PimObjId src, unsigned ofst);
  bool pimOpWriteSaToRow(PimObjId src, unsigned ofst);
  bool pimOpTRA(PimObjId src1, unsigned ofst1, PimObjId src2, unsigned ofst2, PimObjId src3, unsigned ofst3);
  bool pimOpMove(PimObjId objId, PimRowReg src, PimRowReg dest);
  bool pimOpSet(PimObjId objId, PimRowReg dest, bool val);
  bool pimOpNot(PimObjId objId, PimRowReg src, PimRowReg dest);
  bool pimOpAnd(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  bool pimOpOr(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  bool pimOpNand(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  bool pimOpNor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  bool pimOpXor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  bool pimOpXnor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  bool pimOpMaj(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg src3, PimRowReg dest);
  bool pimOpSel(PimObjId objId, PimRowReg cond, PimRowReg src1, PimRowReg src2, PimRowReg dest);
  bool pimOpRotateRH(PimObjId objId, PimRowReg src);
  bool pimOpRotateLH(PimObjId objId, PimRowReg src);

  // SIMDRAM micro ops
  bool pimOpAP(int numSrc, va_list args);
  bool pimOpAAP(int numSrc, int numDest, va_list args);

private:
  pimSim();
  ~pimSim();
  pimSim(const pimSim&) = delete;
  pimSim operator=(const pimSim&) = delete;

  static pimSim* s_instance;

  // support one device for now
  pimDevice* m_device = nullptr;
  pimParamsDram* m_paramsDram = nullptr;
  pimParamsPerf* m_paramsPerf = nullptr;
  pimStatsMgr* m_statsMgr = nullptr;
  pimUtils::threadPool* m_threadPool = nullptr;
  unsigned m_numThreads = 1;
};

#endif

