// File: libpimsim.cpp
// PIM Functional Simulator Library Interface
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "libpimsim.h"
#include "pimSim.h"


//! @brief  Create a PIM device
PimStatus
pimCreateDevice(PimDeviceEnum deviceType, unsigned numCores, unsigned numRows, unsigned numCols)
{
  bool ok = pimSim::get()->createDevice(deviceType, numCores, numRows, numCols);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Create a PIM device from config file
PimStatus
pimCreateDeviceFromConfig(PimDeviceEnum deviceType, const char* configFileName)
{
  bool ok = pimSim::get()->createDeviceFromConfig(deviceType, configFileName);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Delete a PIM device
PimStatus
pimDeleteDevice()
{
  bool ok = pimSim::get()->deleteDevice();
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Show PIM command stats
void
pimShowStats()
{
  pimSim::get()->showStats();
}

//! @brief  Allocate a PIM resource
PimObjId
pimAlloc(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElements, PimDataType dataType)
{
  return pimSim::get()->pimAlloc(allocType, numElements, bitsPerElements, dataType);
}

//! @brief  Allocate a PIM resource, with an associated object as reference
PimObjId
pimAllocAssociated(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElements, PimObjId ref, PimDataType dataType)
{
  return pimSim::get()->pimAllocAssociated(allocType, numElements, bitsPerElements, ref, dataType);
}

//! @brief  Free a PIM resource
PimStatus
pimFree(PimObjId obj)
{
  bool ok = pimSim::get()->pimFree(obj);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Create a obj referencing to a range of an existing obj
PimObjId
pimRangedRef(PimObjId ref, unsigned idxBegin, unsigned idxEnd)
{
  bool ok = false;
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Copy data from main to PIM device
PimStatus
pimCopyHostToDevice(PimCopyEnum copyType, void* src, PimObjId dest)
{
  bool ok = pimSim::get()->pimCopyMainToDevice(copyType, src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Copy data from PIM device to main
PimStatus
pimCopyDeviceToHost(PimCopyEnum copyType, PimObjId src, void* dest)
{
  bool ok = pimSim::get()->pimCopyDeviceToMain(copyType, src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Load vector with a scalar value
PimStatus
pimBroadCast(PimCopyEnum copyType, PimObjId dest, unsigned value)
{
  bool ok = pimSim::get()->pimBroadCast(copyType, dest, value);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM add
PimStatus
pimAdd(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimAdd(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM sub
PimStatus
pimSub(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimSub(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM div
PimStatus
pimDiv(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimDiv(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM or
PimStatus
pimOr(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimOr(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM and
PimStatus
pimAnd(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimAnd(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM xor
PimStatus
pimXor(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimXor(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM abs
PimStatus
pimAbs(PimObjId src, PimObjId dest)
{
  bool ok = pimSim::get()->pimAbs(src, dest);;
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM multiplication
PimStatus
pimMul(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimMul(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM GT
PimStatus
pimGT(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimGT(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM LT
PimStatus
pimLT(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimLT(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM EQ
PimStatus
pimEQ(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimEQ(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM Min
PimStatus
pimMin(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimMin(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM Max
PimStatus
pimMax(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimMax(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM Pop Count
PimStatus
pimPopCount(PimObjId src, PimObjId dest)
{
  bool ok = pimSim::get()->pimPopCount(src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM reduction sum. Result returned to a host variable
int
pimRedSum(PimObjId src)
{
  return pimSim::get()->pimRedSum(src);
}

//! @brief  PIM reduction sum for a range of an obj. Result returned to a host variable
int
pimRedSumRanged(PimObjId src, unsigned idxBegin, unsigned idxEnd)
{
  return pimSim::get()->pimRedSumRanged(src, idxBegin, idxEnd);
}



//! @brief  Rotate all elements of an obj by one step to the right
PimStatus
pimRotateR(PimObjId src)
{
  bool ok = pimSim::get()->pimRotateR(src);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Rotate all elements of an obj by one step to the left
PimStatus
pimRotateL(PimObjId src)
{
  bool ok = pimSim::get()->pimRotateL(src);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Read a row to SA
PimStatus
pimOpReadRowToSa(PimObjId src, unsigned ofst)
{
  bool ok = pimSim::get()->pimOpReadRowToSa(src, ofst);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Write SA to a row
PimStatus
pimOpWriteSaToRow(PimObjId src, unsigned ofst)
{
  bool ok = pimSim::get()->pimOpWriteSaToRow(src, ofst);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Triple row activation to SA
PimStatus
pimOpTRA(PimObjId src1, unsigned ofst1, PimObjId src2, unsigned ofst2, PimObjId src3, unsigned ofst3)
{
  bool ok = pimSim::get()->pimOpTRA(src1, ofst1, src2, ofst2, src3, ofst3);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Move value between two regs
PimStatus
pimOpMove(PimObjId objId, PimRowReg src, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpMove(objId, src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Set value of a reg
PimStatus
pimOpSet(PimObjId objId, PimRowReg src, bool val)
{
  bool ok = pimSim::get()->pimOpSet(objId, src, val);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Not of a reg
PimStatus
pimOpNot(PimObjId objId, PimRowReg src, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpNot(objId, src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: And of two regs
PimStatus
pimOpAnd(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpAnd(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Or of two regs
PimStatus
pimOpOr(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpOr(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Nand of two regs
PimStatus
pimOpNand(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpNand(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Nor of two regs
PimStatus
pimOpNor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpNor(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Xor of two regs
PimStatus
pimOpXor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpXor(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Xnor of two regs
PimStatus
pimOpXnor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpXnor(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Maj of three regs
PimStatus
pimOpMaj(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg src3, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpMaj(objId, src1, src2, src3, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Conditional selecion: dest = cond ? src1 : src2
PimStatus
pimOpSel(PimObjId objId, PimRowReg cond, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpSel(objId, cond, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Rotate a reg to the right, using srcId for range
PimStatus
pimOpRotateRH(PimObjId objId, PimRowReg src)
{
  bool ok = pimSim::get()->pimOpRotateRH(objId, src);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Rotate a reg to the left, using srcId for range
PimStatus
pimOpRotateLH(PimObjId objId, PimRowReg src)
{
  bool ok = pimSim::get()->pimOpRotateLH(objId, src);
  return ok ? PIM_OK : PIM_ERROR;
}

