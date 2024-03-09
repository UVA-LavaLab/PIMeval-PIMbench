// File: pimResMgr.cpp
// PIM Functional Simulator - PIM Resource Manager
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimResMgr.h"


//! @brief  Alloc a PIM object
PimObjId
pimResMgr::pimAlloc(PimAllocEnum allocType, int numElements, int bitsPerElement)
{
  return -1;
}

//! @brief  Alloc a PIM object assiciated to a reference object
PimObjId
pimResMgr::pimAllocAssociated(PimAllocEnum allocType, int numElements, int bitsPerElement, PimObjId ref)
{
  return -1;
}

//! @brief  Free a PIM object
bool
pimResMgr::pimFree(PimObjId obj)
{
  return false;
}

