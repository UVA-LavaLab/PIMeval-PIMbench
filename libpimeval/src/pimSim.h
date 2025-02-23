// File: pimSim.h
// PIMeval Simulator - PIM Simulator Main Entry
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_SIM_H
#define LAVA_PIM_SIM_H

#include "libpimeval.h"
#include "pimSimConfig.h"
#include "pimDevice.h"
#include "pimParamsDram.h"
#include "pimPerfEnergyBase.h"
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
  bool createDevice(PimDeviceEnum deviceType, unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank, unsigned numRows, unsigned numCols, bool isLoadBalanced);
  bool createDeviceFromConfig(PimDeviceEnum deviceType, const char* configFileName);
  bool getDeviceProperties(PimDeviceProperties* deviceProperties);
  bool deleteDevice();
  bool isValidDevice(bool showMsg = true) const;

  // From pimSimConfig
  const pimSimConfig& getConfig() const { return m_config; }
  bool isAnalysisMode() const { return m_config.getAnalysisMode(); }
  unsigned getNumThreads() const { return m_config.getNumThreads(); }
  PimDeviceEnum getDeviceType() const { return m_config.getDeviceType(); }
  PimDeviceEnum getSimTarget() const { return m_config.getSimTarget(); }
  unsigned getNumRanks() const { return m_config.getNumRanks(); }
  unsigned getNumBankPerRank() const { return m_config.getNumBankPerRank(); }
  unsigned getNumSubarrayPerBank() const { return m_config.getNumSubarrayPerBank(); }
  unsigned getNumRowPerSubarray() const { return m_config.getNumRowPerSubarray(); }
  unsigned getNumColPerSubarray() const { return m_config.getNumColPerSubarray(); }

  unsigned getNumCores() const;
  unsigned getNumRows() const;
  unsigned getNumCols() const;

  void startKernelTimer() const;
  void endKernelTimer() const;
  void showStats() const;
  void resetStats() const;
  pimStatsMgr* getStatsMgr() { return m_statsMgr.get(); }
  const pimParamsDram& getParamsDram() const { assert(m_paramsDram); return *m_paramsDram; }
  pimPerfEnergyBase* getPerfEnergyModel();

  pimUtils::threadPool* getThreadPool() { return m_threadPool.get(); }

  // Resource allocation and deletion
  PimObjId pimAlloc(PimAllocEnum allocType, uint64_t numElements, PimDataType dataType);
  PimObjId pimAllocAssociated(PimObjId assocId, PimDataType dataType);
  bool pimFree(PimObjId obj);
  PimObjId pimCreateRangedRef(PimObjId refId, uint64_t idxBegin, uint64_t idxEnd);
  PimObjId pimCreateDualContactRef(PimObjId refId);

  // Data transfer
  bool pimCopyMainToDevice(void* src, PimObjId dest, uint64_t idxBegin = 0, uint64_t idxEnd = 0);
  bool pimCopyDeviceToMain(PimObjId src, void* dest, uint64_t idxBegin = 0, uint64_t idxEnd = 0);
  bool pimCopyMainToDeviceWithType(PimCopyEnum copyType, void* src, PimObjId dest, uint64_t idxBegin = 0, uint64_t idxEnd = 0);
  bool pimCopyDeviceToMainWithType(PimCopyEnum copyType, PimObjId src, void* dest, uint64_t idxBegin = 0, uint64_t idxEnd = 0);
  bool pimCopyDeviceToDevice(PimObjId src, PimObjId dest, uint64_t idxBegin = 0, uint64_t idxEnd = 0);
  bool pimCopyObjectToObject(PimObjId src, PimObjId dest);
  bool pimConvertType(PimObjId src, PimObjId dest);

  // Computation
  bool pimAdd(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimSub(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimDiv(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimAbs(PimObjId src, PimObjId dest);
  bool pimMul(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimNot(PimObjId src, PimObjId dest);
  bool pimOr(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimAnd(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimXor(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimXnor(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimGT(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimLT(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimEQ(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimNE(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimMin(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimMax(PimObjId src1, PimObjId src2, PimObjId dest);
  bool pimAdd(PimObjId src, PimObjId dest, uint64_t scalarValue);
  bool pimSub(PimObjId src, PimObjId dest, uint64_t scalarValue);
  bool pimMul(PimObjId src, PimObjId dest, uint64_t scalarValue);
  bool pimDiv(PimObjId src, PimObjId dest, uint64_t scalarValue);
  bool pimAnd(PimObjId src, PimObjId dest, uint64_t scalarValue);
  bool pimOr(PimObjId src, PimObjId dest, uint64_t scalarValue);
  bool pimXor(PimObjId src, PimObjId dest, uint64_t scalarValue);
  bool pimXnor(PimObjId src, PimObjId dest, uint64_t scalarValue);
  bool pimGT(PimObjId src, PimObjId dest, uint64_t scalarValue);
  bool pimLT(PimObjId src, PimObjId dest, uint64_t scalarValue);
  bool pimEQ(PimObjId src, PimObjId dest, uint64_t scalarValue);
  bool pimNE(PimObjId src, PimObjId dest, uint64_t scalarValue);
  bool pimMin(PimObjId src, PimObjId dest, uint64_t scalarValue);
  bool pimMax(PimObjId src, PimObjId dest, uint64_t scalarValue);
  bool pimScaledAdd(PimObjId src1, PimObjId src2, PimObjId dest, uint64_t scalarValue);
  bool pimPopCount(PimObjId src, PimObjId dest);
  bool pimRedSum(PimObjId src, void* sum, uint64_t idxBegin = 0, uint64_t idxEnd = 0);
  bool pimRedMin(PimObjId src, void* min, uint64_t idxBegin = 0, uint64_t idxEnd = 0);
  bool pimRedMax(PimObjId src, void* max, uint64_t idxBegin = 0, uint64_t idxEnd = 0);
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
  bool createDeviceCommon();
  void uninit();

  static pimSim* s_instance;
  pimSimConfig m_config;

  // support one device for now
  std::unique_ptr<pimDevice> m_device;
  std::unique_ptr<pimParamsDram> m_paramsDram;
  std::unique_ptr<pimStatsMgr> m_statsMgr;
  std::unique_ptr<pimUtils::threadPool> m_threadPool;

};

#endif

