// File: pimResMgr.h
// PIMeval Simulator - PIM Resource Manager
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_RES_MGR_H
#define LAVA_PIM_RES_MGR_H

#include "libpimeval.h"
#include <vector>
#include <tuple>
#include <unordered_map>
#include <set>
#include <map>
#include <string>
#include <memory>
#include <cassert>

class pimDevice;


//! @class  pimRegion
//! @brief  Represent a rectangle regreion in a PIM core
class pimRegion
{
public:
  pimRegion() {}
  ~pimRegion() {}

  void setCoreId(PimCoreId coreId) { m_coreId = coreId; }
  void setRowIdx(unsigned rowIdx) { m_rowIdx = rowIdx; }
  void setColIdx(unsigned colIdx) { m_colIdx = colIdx; }
  void setNumAllocRows(unsigned numAllocRows) { m_numAllocRows = numAllocRows; }
  void setNumAllocCols(unsigned numAllocCols) { m_numAllocCols = numAllocCols; }
  void setElemIdxBegin(uint64_t idx) { m_elemIdxBegin = idx; }
  void setElemIdxEnd(uint64_t idx) { m_elemIdxEnd = idx; }
  void setIsValid(bool val) { m_isValid = val; }
  void setNumColsPerElem(unsigned val) { m_numColsPerElem = val; }

  PimCoreId getCoreId() const { return m_coreId; }
  unsigned getRowIdx() const { return m_rowIdx; }
  unsigned getColIdx() const { return m_colIdx; }
  unsigned getNumAllocRows() const { return m_numAllocRows; }
  unsigned getNumAllocCols() const { return m_numAllocCols; }
  uint64_t getElemIdxBegin() const { return m_elemIdxBegin; }
  uint64_t getElemIdxEnd() const { return m_elemIdxEnd; }
  uint64_t getNumElemInRegion() const { return m_elemIdxEnd - m_elemIdxBegin; }
  unsigned getNumColsPerElem() const { return m_numColsPerElem; }

  std::pair<unsigned, unsigned> locateIthElemInRegion(unsigned i) const {
    assert(i < getNumElemInRegion());
    unsigned rowIdx = m_rowIdx; // only one row of elements per region
    unsigned colIdx = m_colIdx + i * m_numColsPerElem;
    return std::make_pair(rowIdx, colIdx);
  }

  bool isValid() const { return m_isValid && m_coreId >= 0 && m_numAllocRows > 0 && m_numAllocCols > 0; }

  void print(uint64_t regionId) const;

private:
  PimCoreId m_coreId = -1;
  unsigned m_rowIdx = 0;        // starting row index
  unsigned m_colIdx = 0;        // starting col index
  unsigned m_numAllocRows = 0;  // number of rows of this region
  unsigned m_numAllocCols = 0;  // number of cols of this region
  uint64_t m_elemIdxBegin = 0;  // begin element index in this region
  uint64_t m_elemIdxEnd = 0;    // end element index in this region
  unsigned m_numColsPerElem = 0;  // number of cols per element
  bool m_isValid = false;
};


//! @class  pimObjInfo
//! @brief  Meta data of a PIM object which includes
//!         - PIM object ID
//!         - One or more rectangle regions allocated in one or more PIM cores
//!         - Allocation type which specifies how data is stored in a region
class pimObjInfo
{
public:
  pimObjInfo(PimObjId objId, PimDataType dataType, PimAllocEnum allocType, uint64_t numElements, unsigned bitsPerElement)
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
  void setNumColsPerElem(unsigned val) { m_numColsPerElem = val; }
  void finalize();

  PimObjId getObjId() const { return m_objId; }
  PimObjId getAssocObjId() const { return m_assocObjId; }
  PimObjId getRefObjId() const { return m_refObjId; }
  bool isDualContactRef() const { return m_isDualContactRef; }
  PimAllocEnum getAllocType() const { return m_allocType; }
  PimDataType getDataType() const { return m_dataType; }
  uint64_t getNumElements() const { return m_numElements; }
  unsigned getBitsPerElement() const { return m_bitsPerElement; }
  unsigned getBitsPerElementPadded() const { return m_bitsPerElementPadded; }
  bool isValid() const { return m_numElements > 0 && m_bitsPerElement > 0 && !m_regions.empty(); }
  bool isVLayout() const { return m_allocType == PIM_ALLOC_V || m_allocType == PIM_ALLOC_V1; }
  bool isHLayout() const { return m_allocType == PIM_ALLOC_H || m_allocType == PIM_ALLOC_H1; }

  const std::vector<pimRegion>& getRegions() const { return m_regions; }
  std::vector<pimRegion> getRegionsOfCore(PimCoreId coreId) const;
  unsigned getMaxNumRegionsPerCore() const { return m_maxNumRegionsPerCore; }
  unsigned getNumCoresUsed() const { return m_numCoresUsed; }
  unsigned getMaxElementsPerRegion() const { return m_maxElementsPerRegion; }
  unsigned getNumColsPerElem() const { return m_numColsPerElem; }

  std::string getDataTypeName() const;
  void print() const;

private:
  PimObjId m_objId = -1;
  PimObjId m_assocObjId = -1;
  PimObjId m_refObjId = -1;
  PimDataType m_dataType;
  PimAllocEnum m_allocType;
  uint64_t m_numElements = 0;
  unsigned m_bitsPerElement = 0;
  unsigned m_bitsPerElementPadded = 0;
  std::vector<pimRegion> m_regions;  // a list of core ID and regions
  unsigned m_maxNumRegionsPerCore = 0;
  unsigned m_numCoresUsed = 0;
  unsigned m_maxElementsPerRegion = 0;
  unsigned m_numColsPerElem = 0; // number of cols per element
  bool m_isDualContactRef = false;
};


//! @class  pimResMgr
//! @brief  PIM resource manager
class pimResMgr
{
public:
  pimResMgr(pimDevice* device);
  ~pimResMgr();

  PimObjId pimAlloc(PimAllocEnum allocType, uint64_t numElements, unsigned bitsPerElement, PimDataType dataType);
  PimObjId pimAllocAssociated(unsigned bitsPerElement, PimObjId assocId, PimDataType dataType);
  bool pimFree(PimObjId objId);
  PimObjId pimCreateRangedRef(PimObjId refId, uint64_t idxBegin, uint64_t idxEnd);
  PimObjId pimCreateDualContactRef(PimObjId refId);

  bool isValidObjId(PimObjId objId) const { return m_objMap.find(objId) != m_objMap.end(); }
  const pimObjInfo& getObjInfo(PimObjId objId) const { return m_objMap.at(objId); }

  bool isVLayoutObj(PimObjId objId) const;
  bool isHLayoutObj(PimObjId objId) const;
  bool isHybridLayoutObj(PimObjId objId) const;

private:
  pimRegion findAvailRegionOnCore(PimCoreId coreId, unsigned numAllocRows, unsigned numAllocCols) const;
  std::vector<PimCoreId> getCoreIdsSortedByLeastUsage() const;

  //! @class  coreUsage
  //! @brief  Track row usage for allocation
  class coreUsage {
  public:
    coreUsage(unsigned numRowsPerCore) : m_numRowsPerCore(numRowsPerCore) {}
    ~coreUsage() {}
    unsigned getNumRowsPerCore() const { return m_numRowsPerCore; }
    unsigned getTotRowsInUse() const { return m_totRowsInUse; }
    unsigned findAvailRange(unsigned numRowsToAlloc);
    void addRange(std::pair<unsigned, unsigned> range, PimObjId objId);
    void deleteObj(PimObjId objId);
    void newAllocStart();
    void newAllocEnd(bool success);
  private:
    unsigned m_numRowsPerCore = 0;
    unsigned m_totRowsInUse = 0;
    std::map<std::pair<unsigned, unsigned>, PimObjId> m_rangesInUse;
    std::set<std::pair<unsigned, unsigned>> m_newAlloc;
  };

  pimDevice* m_device;
  PimObjId m_availObjId;
  std::unordered_map<PimObjId, pimObjInfo> m_objMap;
  std::unordered_map<PimCoreId, std::unique_ptr<pimResMgr::coreUsage>> m_coreUsage;
  std::unordered_map<PimObjId, std::set<PimObjId>> m_refMap;
};

#endif

