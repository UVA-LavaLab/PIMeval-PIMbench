// File: pimResMgr.h
// PIM Functional Simulator - PIM Resource Manager
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_RES_MGR_H
#define LAVA_PIM_RES_MGR_H

#include "libpimsim.h"
#include <vector>
#include <tuple>
#include <unordered_map>
#include <set>
#include <string>

class pimDevice;


//! @class  pimRegion
//! @brief  Represent a rectangle regreion in a PIM core
class pimRegion
{
public:
  pimRegion()
    : m_coreId(-1),
      m_rowIdx(0),
      m_colIdx(0),
      m_numAllocRows(0),
      m_numAllocCols(0),
      m_isValid(false)
  {}
  ~pimRegion() {}

  void setCoreId(PimCoreId coreId) { m_coreId = coreId; }
  void setRowIdx(unsigned rowIdx) { m_rowIdx = rowIdx; }
  void setColIdx(unsigned colIdx) { m_colIdx = colIdx; }
  void setNumAllocRows(unsigned numAllocRows) { m_numAllocRows = numAllocRows; }
  void setNumAllocCols(unsigned numAllocCols) { m_numAllocCols = numAllocCols; }
  void setIsValid(bool val) { m_isValid = val; }

  PimCoreId getCoreId() const { return m_coreId; }
  unsigned getRowIdx() const { return m_rowIdx; }
  unsigned getColIdx() const { return m_colIdx; }
  unsigned getNumAllocRows() const { return m_numAllocRows; }
  unsigned getNumAllocCols() const { return m_numAllocCols; }

  bool isValid() const { return m_isValid && m_coreId >= 0 && m_numAllocRows > 0 && m_numAllocCols > 0; }

  void print() const;

private:
  PimCoreId m_coreId;
  unsigned m_rowIdx;  // starting row index
  unsigned m_colIdx;  // starting col index
  unsigned m_numAllocRows;  // number of rows of this region
  unsigned m_numAllocCols;  // number of cols of this region
  bool m_isValid;
};


//! @class  pimObjInfo
//! @brief  Meta data of a PIM object which includes
//!         - PIM object ID
//!         - One or more rectangle regions allocated in one or more PIM cores
//!         - Allocation type which specifies how data is stored in a region
class pimObjInfo
{
public:
  pimObjInfo(PimObjId objId, PimDataType dataType, PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement)
    : m_objId(objId),
      m_assocObjId(objId),
      m_dataType(dataType),
      m_allocType(allocType),
      m_numElements(numElements),
      m_bitsPerElement(bitsPerElement)
  {}
  ~pimObjInfo() {}

  void addRegion(pimRegion region) { m_regions.push_back(region); }
  void setObjId(PimObjId objId) { m_objId = objId; }
  void setAssocObjId(PimObjId assocObjId) { m_assocObjId = assocObjId; }
  void setRefObjId(PimObjId refObjId) { m_refObjId = refObjId; }
  void setIsDualContactRef(bool val) { m_isDualContactRef = val; }
  void finalize();

  PimObjId getObjId() const { return m_objId; }
  PimObjId getAssocObjId() const { return m_assocObjId; }
  PimObjId getRefObjId() const { return m_refObjId; }
  bool isDualContactRef() const { return m_isDualContactRef; }
  PimAllocEnum getAllocType() const { return m_allocType; }
  PimDataType getDataType() const { return m_dataType; }
  unsigned getNumElements() const { return m_numElements; }
  unsigned getBitsPerElement() const { return m_bitsPerElement; }
  bool isValid() const { return m_numElements > 0 && m_bitsPerElement > 0 && !m_regions.empty(); }
  bool isVLayout() const { return m_allocType == PIM_ALLOC_V || m_allocType == PIM_ALLOC_V1; }
  bool isHLayout() const { return m_allocType == PIM_ALLOC_H || m_allocType == PIM_ALLOC_H1; }

  const std::vector<pimRegion>& getRegions() const { return m_regions; }
  std::vector<pimRegion> getRegionsOfCore(PimCoreId coreId) const;
  unsigned getMaxNumRegionsPerCore() const { return m_maxNumRegionsPerCore; }
  unsigned getNumCoresUsed() const { return m_numCoresUsed; }
  unsigned getMaxElementsPerRegion() const { return m_maxElementsPerRegion; }

  std::string getDataTypeName() const;
  void print() const;

private:
  PimObjId m_objId = -1;
  PimObjId m_assocObjId = -1;
  PimObjId m_refObjId = -1;
  PimDataType m_dataType;
  PimAllocEnum m_allocType;
  unsigned m_numElements = 0;
  unsigned m_bitsPerElement = 0;
  std::vector<pimRegion> m_regions;  // a list of core ID and regions
  unsigned m_maxNumRegionsPerCore = 0;
  unsigned m_numCoresUsed = 0;
  unsigned m_maxElementsPerRegion = 0;
  bool m_isDualContactRef = false;
};


//! @class  pimResMgr
//! @brief  PIM resource manager
class pimResMgr
{
public:
  pimResMgr(pimDevice* device)
    : m_device(device),
      m_availObjId(0)
  {}
  ~pimResMgr() {}

  PimObjId pimAlloc(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement, PimDataType dataType);
  PimObjId pimAllocAssociated(unsigned bitsPerElement, PimObjId assocId, PimDataType dataType);
  bool pimFree(PimObjId objId);
  PimObjId pimCreateRangedRef(PimObjId refId, unsigned idxBegin, unsigned idxEnd);
  PimObjId pimCreateDualContactRef(PimObjId refId);

  bool isValidObjId(PimObjId objId) const { return m_objMap.find(objId) != m_objMap.end(); }
  const pimObjInfo& getObjInfo(PimObjId objId) const { return m_objMap.at(objId); }

  bool isVLayoutObj(PimObjId objId) const;
  bool isHLayoutObj(PimObjId objId) const;
  bool isHybridLayoutObj(PimObjId objId) const;

private:
  pimRegion findAvailRegionOnCore(PimCoreId coreId, unsigned numAllocRows, unsigned numAllocCols) const;
  std::vector<PimCoreId> getCoreIdsSortedByLeastUsage() const;
  unsigned getCoreUsage(PimCoreId coreId) const;

  pimDevice* m_device;
  PimObjId m_availObjId;
  std::unordered_map<PimObjId, pimObjInfo> m_objMap;
  std::unordered_map<PimCoreId, std::set<std::pair<unsigned, unsigned>>> m_coreUsage; // track row usage only for now
  std::unordered_map<PimObjId, std::set<PimObjId>> m_refMap;
};

#endif

