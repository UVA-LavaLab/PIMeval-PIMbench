// File: pimResMgr.h
// PIM Functional Simulator - PIM Resource Manager
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_RES_MGR_H
#define LAVA_PIM_RES_MGR_H

#include "libpimsim.h"
#include <vector>
#include <tuple>
#include <unordered_map>

//! @class  pimObj
//! @brief  Resource meta data of a PIM object
class pimObj
{
public:
  pimObj() {}
  ~pimObj() {}
private: 
};

//! @class  pimResMgr
//! @brief  PIM resource manager
class pimResMgr
{
public:
  pimResMgr() : m_availObjId(0) {}
  ~pimResMgr() {}

  PimObjId pimAlloc(PimAllocEnum allocType, int numElements, int bitsPerElement);
  PimObjId pimAllocAssociated(PimAllocEnum allocType, int numElements, int bitsPerElement, PimObjId ref);
  bool pimFree(PimObjId obj);

private:
  PimObjId m_availObjId;

  std::unordered_map<PimObjId, pimObj> m_res;
};

#endif

