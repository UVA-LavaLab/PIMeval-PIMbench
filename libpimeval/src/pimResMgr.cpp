// File: pimResMgr.cpp
// PIMeval Simulator - PIM Resource Manager
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "pimResMgr.h"
#include "pimDevice.h"
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <memory>


//! @brief  Print info of a PIM region
void
pimRegion::print(uint64_t regionId) const
{
  #if defined(DEBUG)
  std::printf("{ PIM-Region %lld: CoreId = %d, Loc = (%u, %u), Size = (%u, %u) }\n",
              regionId, m_coreId, m_rowIdx, m_colIdx, m_numAllocRows, m_numAllocCols);
  #endif
}

//! @brief  Print info of a PIM object
void
pimObjInfo::print() const
{
  #if defined(DEBUG)
  std::printf("----------------------------------------\n");
  std::printf("PIM-Object: ObjId = %d, AllocType = %d, Regions =\n",
              m_objId, static_cast<int>(m_allocType));
  for (size_t i = 0; i < m_regions.size(); ++i) {
    m_regions[i].print(i);
  }
  std::printf("----------------------------------------\n");
  #endif
}

std::string
pimObjInfo::getDataTypeName() const
{
  switch (m_dataType)
  {
  case PimDataType::PIM_INT32:
    return "int32";
  case PimDataType::PIM_INT64:
    return "int64";
  default:
    throw std::invalid_argument("Unsupported Type.");
  }
}

//! @brief  Finalize obj info
void
pimObjInfo::finalize()
{
  std::unordered_map<PimCoreId, int> coreIdCnt;
  for (const auto& region : m_regions) {
    PimCoreId coreId = region.getCoreId();
    coreIdCnt[coreId]++;
    unsigned numRegionsPerCore = coreIdCnt[coreId];
    if (m_maxNumRegionsPerCore < numRegionsPerCore) {
      m_maxNumRegionsPerCore = numRegionsPerCore;
    }
  }
  m_numCoresUsed = coreIdCnt.size();

  const pimRegion& region = m_regions[0];
  m_maxElementsPerRegion = (uint64_t)region.getNumAllocRows() * region.getNumAllocCols() / m_bitsPerElement;
  m_numColsPerElem = region.getNumColsPerElem();
}

//! @brief  Get all regions on a specific PIM core for current PIM object
std::vector<pimRegion>
pimObjInfo::getRegionsOfCore(PimCoreId coreId) const
{
  std::vector<pimRegion> regions;
  for (const auto& region : m_regions) {
    if (region.getCoreId() == coreId) {
      regions.push_back(region);
    }
  }
  return regions;
}

//! @brief  pimResMgr ctor
pimResMgr::pimResMgr(pimDevice* device)
  : m_device(device),
    m_availObjId(0)
{
  unsigned numCores = m_device->getNumCores();
  unsigned numRowsPerCore = m_device->getNumRows();
  for (unsigned i = 0; i < numCores; ++i) {
    m_coreUsage[i] = std::make_unique<coreUsage>(numRowsPerCore);
  }
}

//! @brief  pimResMgr dtor
pimResMgr::~pimResMgr()
{
}

//! @brief  Alloc a PIM object
//!         Expect bitsPerElement to be smaller than or equal to the width of dataType
//!         For V layout, allocate bitsPerElement rows
//!         For H layout, pad element to the width of dataType
PimObjId
pimResMgr::pimAlloc(PimAllocEnum allocType, uint64_t numElements, unsigned bitsPerElement, PimDataType dataType)
{
  #if defined(DEBUG)
  std::printf("PIM-Debug: pimResMgr::pimAlloc for %d alloc-type %llu elements %u bits per element %d data-type\n",
              (int)allocType, numElements, bitsPerElement, (int)dataType);
  #endif

  if (numElements == 0 || bitsPerElement == 0) {
    std::printf("PIM-Error: Invalid parameters to allocate %llu elements of %u bits\n", numElements, bitsPerElement);
    return -1;
  }
  unsigned bitsOfDataType = pimUtils::getNumBitsOfDataType(dataType);
  if (bitsPerElement > bitsOfDataType) {
    std::printf("PIM-Error: Cannot allocate %u bits which is greater than the width of %s data type\n",
                bitsPerElement, pimUtils::pimDataTypeEnumToStr(dataType).c_str());
    return -1;
  }

  std::vector<PimCoreId> sortedCoreId = getCoreIdsSortedByLeastUsage();
  pimObjInfo newObj(m_availObjId, dataType, allocType, numElements, bitsPerElement);
  m_availObjId++;

  unsigned numCores = m_device->getNumCores();
  unsigned numCols = m_device->getNumCols();
  unsigned numRowsToAlloc = 0;
  uint64_t numRegions = 0;
  unsigned numColsToAllocLast = 0;
  uint64_t numElemPerRegion = 0;
  uint64_t numElemPerRegionLast = 0;
  unsigned numColsPerElem = 0;
  if (allocType == PIM_ALLOC_V || allocType == PIM_ALLOC_V1) {
    // allocate one region per core, with vertical layout
    numRowsToAlloc = bitsPerElement;
    numRegions = (numElements - 1) / numCols + 1;
    numColsToAllocLast = numElements % numCols;
    if (numColsToAllocLast == 0) {
      numColsToAllocLast = numCols;
    }
    numElemPerRegion = numCols;
    numElemPerRegionLast = numColsToAllocLast;
    numColsPerElem = 1;
  } else if (allocType == PIM_ALLOC_H || allocType == PIM_ALLOC_H1) {
    // allocate one region per core, with horizontal layout
    numRowsToAlloc = 1;
    numRegions = (numElements * bitsOfDataType - 1) / numCols + 1;
    numColsToAllocLast = (numElements * bitsOfDataType) % numCols;
    if (numColsToAllocLast == 0) {
      numColsToAllocLast = numCols;
    }
    numElemPerRegion = numCols / bitsPerElement;
    numElemPerRegionLast = numColsToAllocLast / bitsPerElement;
    numColsPerElem = bitsPerElement;
  } else {
    std::printf("PIM-Error: Unsupported PIM allocation type %d\n", static_cast<int>(allocType));
    return -1;
  }

  if (numRegions > numCores) {
    if (allocType == PIM_ALLOC_V1 || allocType == PIM_ALLOC_H1) {
      std::printf("PIM-Error: Obj requires %llu regions among %u cores. Abort since wrapping up is disabled.\n",
                  numRegions, numCores);
      return -1;
    } else {
      #if defined(DEBUG)
      std::printf("PIM-Warning: Obj requires %llu regions among %u cores. Wrapping up is needed.\n", numRegions, numCores);
      #endif
    }
  }

  // create regions
  bool success = true;
  for (unsigned i = 0; i < numCores; ++i) {
    m_coreUsage.at(i)->newAllocStart();
  }
  if (allocType == PIM_ALLOC_V || allocType == PIM_ALLOC_V1 || allocType == PIM_ALLOC_H || allocType == PIM_ALLOC_H1) {
    uint64_t elemIdx = 0;
    for (uint64_t i = 0; i < numRegions; ++i) {
      PimCoreId coreId = sortedCoreId[i % numCores];
      unsigned numColsToAlloc = (i == numRegions - 1 ? numColsToAllocLast : numCols);
      unsigned numElemInRegion = (i == numRegions - 1 ? numElemPerRegionLast : numElemPerRegion);
      pimRegion newRegion = findAvailRegionOnCore(coreId, numRowsToAlloc, numColsToAlloc);
      if (!newRegion.isValid()) {
        std::printf("PIM-Error: Failed to allocate object with %u rows on core %d\n", numRowsToAlloc, coreId);
        success = false;
        break;
      }
      newRegion.setElemIdxBegin(elemIdx);
      elemIdx += numElemInRegion;
      newRegion.setElemIdxEnd(elemIdx); // exclusive
      newRegion.setNumColsPerElem(numColsPerElem);
      newObj.addRegion(newRegion);

      // add to core usage map
      auto alloc = std::make_pair(newRegion.getRowIdx(), numRowsToAlloc);
      m_coreUsage.at(coreId)->addRange(alloc, newObj.getObjId());
    }
  }
  for (unsigned i = 0; i < numCores; ++i) {
    m_coreUsage.at(i)->newAllocEnd(success); // rollback if failed
  }

  if (!success) {
    return -1;
  }

  PimObjId objId = -1;
  if (newObj.isValid()) {
    objId = newObj.getObjId();
    newObj.finalize();
    newObj.print();
    // update new object to resource mgr
    m_objMap.insert(std::make_pair(newObj.getObjId(), newObj));
  }

  #if defined(DEBUG)
  std::printf("PIM-Debug: pimResMgr::pimAlloc allocated new object %d with %llu regions\n",
              objId, newObj.getRegions().size());
  #endif
  return objId;
}

//! @brief  Alloc a PIM object assiciated to an existing object
//!         Expect bitsPerElement to be smaller than or equal to the width of dataType
//!         Expect dataType to be narrower or equal to the width of associated dataType
//!         For V layout, allocate bitsPerElement rows
//!         For H layout, pad element to the width of assiciated dataType
//!         Number of elements will be the same as the assiciated object
PimObjId
pimResMgr::pimAllocAssociated(unsigned bitsPerElement, PimObjId assocId, PimDataType dataType)
{
  #if defined(DEBUG)
  std::printf("PIM-Debug: pimResMgr::pimAllocAssociated for %u bits per element %d data-type associated to object %d\n",
              bitsPerElement, (int)dataType, assocId);
  #endif

  // check if assoc obj is valid
  if (m_objMap.find(assocId) == m_objMap.end()) {
    std::printf("PIM-Error: Invalid associated object ID %d for PIM allocation\n", assocId);
    return -1;
  }
  unsigned bitsOfDataType = pimUtils::getNumBitsOfDataType(dataType);
  if (bitsPerElement > bitsOfDataType) {
    std::printf("PIM-Error: Cannot allocate %u bits which is greater than the width of %s data type\n",
                bitsPerElement, pimUtils::pimDataTypeEnumToStr(dataType).c_str());
    return -1;
  }

  // get regions of the assoc obj
  unsigned numCores = m_device->getNumCores();
  const pimObjInfo& assocObj = m_objMap.at(assocId);

  // check if the request can be associated with ref
  PimAllocEnum allocType = assocObj.getAllocType();
  uint64_t numElements = assocObj.getNumElements();
  PimDataType assocDataType = assocObj.getDataType();
  unsigned bitsOfAssocDataType = pimUtils::getNumBitsOfDataType(assocDataType);
  if (allocType == PIM_ALLOC_H || allocType == PIM_ALLOC_H1) {
    if (bitsOfDataType > bitsOfAssocDataType) {
      std::printf("PIM-Error: Cannot associate or pad a wider %s data type to %s data type\n",
                  pimUtils::pimDataTypeEnumToStr(dataType).c_str(),
                  pimUtils::pimDataTypeEnumToStr(assocDataType).c_str());
      return -1;
    }
  }
  assert(allocType == assocObj.getAllocType());

  // allocate associated regions
  pimObjInfo newObj(m_availObjId, dataType, allocType, numElements, bitsPerElement);
  m_availObjId++;

  bool success = true;
  for (unsigned i = 0; i < numCores; ++i) {
    m_coreUsage.at(i)->newAllocStart();
  }
  for ( const pimRegion& region : assocObj.getRegions()) {
    PimCoreId coreId = region.getCoreId();
    unsigned numAllocRows = region.getNumAllocRows();
    unsigned numAllocCols = region.getNumAllocCols();
    if (allocType == PIM_ALLOC_V || allocType == PIM_ALLOC_V1) {
      numAllocRows = bitsPerElement;
    }
    pimRegion newRegion = findAvailRegionOnCore(coreId, numAllocRows, numAllocCols);
    if (!newRegion.isValid()) {
      std::printf("PIM-Error: Failed to allocate associated object with %u rows on core %d\n", numAllocRows, coreId);
      success = false;
      break;
    }
    newRegion.setElemIdxBegin(region.getElemIdxBegin());
    newRegion.setElemIdxEnd(region.getElemIdxEnd()); // exclusive
    newRegion.setNumColsPerElem(region.getNumColsPerElem());
    newObj.addRegion(newRegion);

    // add to core usage map
    auto alloc = std::make_pair(newRegion.getRowIdx(), numAllocRows);
    m_coreUsage.at(coreId)->addRange(alloc, newObj.getObjId());
  }
  for (unsigned i = 0; i < numCores; ++i) {
    m_coreUsage.at(i)->newAllocEnd(success); // rollback if failed
  }

  if (!success) {
    return -1;
  }

  PimObjId objId = -1;
  if (newObj.isValid()) {
    objId = newObj.getObjId();
    newObj.finalize();
    newObj.print();
    newObj.setAssocObjId(assocObj.getAssocObjId());
    // update new object to resource mgr
    m_objMap.insert(std::make_pair(newObj.getObjId(), newObj));
  }

  #if defined(DEBUG)
  std::printf("PIM-Debug: pimResMgr::pimAllocAssociated allocated new object %d with %llu regions\n",
              objId, newObj.getRegions().size());
  #endif
  return objId;
}

//! @brief  Free a PIM object
bool
pimResMgr::pimFree(PimObjId objId)
{
  if (m_objMap.find(objId) == m_objMap.end()) {
    std::printf("PIM-Error: Cannot free non-exist object ID %d\n", objId);
    return false;
  }
  unsigned numCores = m_device->getNumCores();
  const pimObjInfo& obj = m_objMap.at(objId);

  if (!obj.isDualContactRef()) {
    for (unsigned i = 0; i < numCores; ++i) {
      m_coreUsage.at(i)->deleteObj(objId);
    }
  }
  m_objMap.erase(objId);

  // free all reference as well
  if (m_refMap.find(objId) != m_refMap.end()) {
    for (auto refId : m_refMap.at(objId)) {
      m_objMap.erase(refId);
    }
  }

  return true;
}

//! @brief  Create an obj referencing to a range of an existing obj
PimObjId
pimResMgr::pimCreateRangedRef(PimObjId refId, uint64_t idxBegin, uint64_t idxEnd)
{
  assert(0); // todo
  return -1;
}

//! @brief  Create an obj referencing to negation of an existing obj based on dual-contact memory cells
PimObjId
pimResMgr::pimCreateDualContactRef(PimObjId refId)
{
  // check if ref obj is valid
  if (m_objMap.find(refId) == m_objMap.end()) {
    std::printf("PIM-Error: Invalid ref object ID %d for PIM dual contact ref\n", refId);
    return -1;
  }

  const pimObjInfo& refObj = m_objMap.at(refId);
  if (refObj.isDualContactRef()) {
    std::printf("PIM-Error: Cannot create dual contact ref of dual contact ref %d\n", refId);
    return -1;
  }

  // The dual-contact ref has exactly same regions as the ref object.
  // The refObjId field points to the ref object.
  // The isDualContactRef field indicates that values need to be negated during read/write.
  pimObjInfo newObj = refObj;
  PimObjId objId = m_availObjId++;
  newObj.setObjId(objId);
  newObj.setRefObjId(refObj.getObjId());
  m_refMap[refObj.getObjId()].insert(objId);
  newObj.setIsDualContactRef(true);
  m_objMap.insert(std::make_pair(newObj.getObjId(), newObj));

  return objId;
}

//! @brief  Alloc resource on a specific core. Perform row allocation for now.
pimRegion
pimResMgr::findAvailRegionOnCore(PimCoreId coreId, unsigned numAllocRows, unsigned numAllocCols) const
{
  pimRegion region;
  region.setCoreId(coreId);
  region.setColIdx(0);
  region.setNumAllocRows(numAllocRows);
  region.setNumAllocCols(numAllocCols);

  // try to find an available slot
  unsigned prevAvail = m_coreUsage.at(coreId)->findAvailRange(numAllocRows);
  if (m_device->getNumRows() - prevAvail >= numAllocRows) {
    region.setRowIdx(prevAvail);
    region.setIsValid(true);
    return region;
  }

  return region;
}

//! @brief  Get a list of core IDs sorted by least usage
std::vector<PimCoreId>
pimResMgr::getCoreIdsSortedByLeastUsage() const
{
  std::vector<std::pair<unsigned, unsigned>> usages;
  for (unsigned coreId = 0; coreId < m_device->getNumCores(); ++coreId) {
    unsigned usage = m_coreUsage.at(coreId)->getTotRowsInUse();
    usages.emplace_back(usage, coreId);
  }
  std::sort(usages.begin(), usages.end());
  std::vector<PimCoreId> result;
  for (const auto& it : usages) {
    result.push_back(it.second);
  }
  return result;
}

//! @brief  Find next available range of rows with a given size
unsigned
pimResMgr::coreUsage::findAvailRange(unsigned numRowsToAlloc)
{
  unsigned prevAvail = 0;
  for (const auto& it : m_rangesInUse) {
    unsigned rowIdx = it.first.first;
    unsigned numRows = it.first.second;
    if (rowIdx - prevAvail >= numRowsToAlloc) {
      return prevAvail;
    }
    prevAvail = rowIdx + numRows;
  }
  return prevAvail;
}

//! @brief  Add a new range to core usage.
//! The new range will be aggregated with previous adjacent ragne if they are from same object
//! Returned range is after aggregation
void
pimResMgr::coreUsage::addRange(std::pair<unsigned, unsigned> range, PimObjId objId)
{
  // aggregate with the prev range
  if (!m_rangesInUse.empty()) {
    auto it = std::prev(m_rangesInUse.end());
    unsigned lastIdx = it->first.first;
    unsigned lastSize = it->first.second;
    PimObjId lastObjId = it->second;
    if (lastIdx + lastSize == range.first && lastObjId == objId) {
      m_newAlloc.erase(it->first);
      m_rangesInUse.erase(it);
      range = std::make_pair(lastIdx, lastSize + range.second);
    }
  }
  m_rangesInUse.insert(std::make_pair(range, objId));
  m_newAlloc.insert(range);
}

//! @brief  Delete an object from core usage
void
pimResMgr::coreUsage::deleteObj(PimObjId objId)
{
  for (auto it = m_rangesInUse.begin(); it != m_rangesInUse.end();) {
    if (it->second == objId) {
      it = m_rangesInUse.erase(it);
    } else {
      ++it;
    }
  }
}

//! @brief  Start a new allocation. This is preparing for rollback
void
pimResMgr::coreUsage::newAllocStart()
{
  m_newAlloc.clear();
}

//! @brief  End a new allocation. If failed, rollback all regions
void
pimResMgr::coreUsage::newAllocEnd(bool success)
{
  if (!success) {
    for (const auto &range : m_newAlloc) {
      m_rangesInUse.erase(range);
    }
  }
  m_newAlloc.clear();
}

//! @brief  If a PIM object uses vertical data layout
bool
pimResMgr::isVLayoutObj(PimObjId objId) const
{
  const pimObjInfo& obj = getObjInfo(objId);
  return obj.isVLayout();
}

//! @brief  If a PIM object uses horizontal data layout
bool
pimResMgr::isHLayoutObj(PimObjId objId) const
{
  const pimObjInfo& obj = getObjInfo(objId);
  return obj.isHLayout();
}

//! @brief  If a PIM object uses hybrid data layout
bool
pimResMgr::isHybridLayoutObj(PimObjId objId) const
{
  return false;
}

