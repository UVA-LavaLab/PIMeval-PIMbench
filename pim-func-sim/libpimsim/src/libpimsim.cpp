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

//! @brief  Allocate a PIM resource
PimObjId
pimAlloc(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElements)
{
  return pimSim::get()->pimAlloc(allocType, numElements, bitsPerElements);
}

//! @brief  Allocate a PIM resource, with an associated object as reference
PimObjId
pimAllocAssociated(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElements, PimObjId ref)
{
  return pimSim::get()->pimAllocAssociated(allocType, numElements, bitsPerElements, ref);
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

//! @brief  PIM int32 vector add
PimStatus
pimInt32Add(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimInt32Add(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM int32 vector abs
PimStatus
pimInt32Abs(PimObjId src, PimObjId dest)
{
  bool ok = false;
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM int32 vector multiplication
PimStatus
pimInt32Mul(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimInt32Mul(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}


//! @brief  PIM int32 reduction sum. Result returned to a host variable
int
pimInt32RedSum(PimObjId src)
{
  return 0;
}

//! @brief  PIM int32 reduction sum for a range of an obj. Result returned to a host variable
int
pimInt32RedSumRanged(PimObjId src, unsigned idxBegin, unsigned idxEnd)
{
  return 0;
}

//! @brief  Rotate all elements of an obj by one step to the right
PimStatus
pimRotateR(PimObjId src)
{
  bool ok = false;
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Rotate all elements of an obj by one step to the left
PimStatus
pimRotateL(PimObjId src)
{
  bool ok = false;
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

