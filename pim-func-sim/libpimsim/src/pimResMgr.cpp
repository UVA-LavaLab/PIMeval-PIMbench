// File: pimResMgr.cpp
// PIM Functional Simulator - PIM Resource Manager
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimResMgr.h"
#include "pimDevice.h"
#include <cstdio>
#include <algorithm>
#include <stdexcept>


//! @brief  Print info of a PIM region
void
pimRegion::print() const
{
  std::printf("{ PIM-Region: CoreId = %d, Loc = (%u, %u), Size = (%u, %u) }\n",
              m_coreId, m_rowIdx, m_colIdx, m_numAllocRows, m_numAllocCols);
}

//! @brief  Print info of a PIM object
void
pimObjInfo::print() const
{
  std::printf("----------------------------------------\n");
  std::printf("PIM-Object: ObjId = %d, AllocType = %d, Regions =\n",
              m_objId, static_cast<int>(m_allocType));
  for (const auto& region : m_regions) {
    region.print();
  }
  std::printf("----------------------------------------\n");
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


//! @brief  Alloc a PIM object
PimObjId
pimResMgr::pimAlloc(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement, PimDataType dataType)
{
  if (numElements <= 0 || bitsPerElement <= 0) {
    std::printf("PIM-Error: Invalid parameters to allocate %u elements of %u bits\n", numElements, bitsPerElement);
    return -1;
  }

  std::vector<PimCoreId> sortedCoreId = getCoreIdsSortedByLeastUsage();
  pimObjInfo newObj(m_availObjId, dataType, allocType, numElements, bitsPerElement);
  m_availObjId++;

  unsigned numCols = m_device->getNumCols();
  unsigned numRowsToAlloc = 0;
  unsigned numRegions = 0;
  unsigned numCoresReqd = 0;
  unsigned numColsToAllocLast = 0;
  if (allocType == PIM_ALLOC_V1) {
    // allocate one region per core, with vertical layout
    numRowsToAlloc = bitsPerElement;
    numRegions = (numElements - 1) / numCols + 1;
    numCoresReqd = numRegions;
    numColsToAllocLast = numElements % numCols;
    if (numColsToAllocLast == 0) {
      numColsToAllocLast = numCols;
    }
  } else if (allocType == PIM_ALLOC_H1) {
    // allocate one region per core, with horizontal layout
    numRowsToAlloc = 1;
    numRegions = (numElements * bitsPerElement - 1) / numCols + 1;
    numCoresReqd = numRegions;
    numColsToAllocLast = (numElements * bitsPerElement) % numCols;
    if (numColsToAllocLast == 0) {
      numColsToAllocLast = numCols;
    }
  } else {
    std::printf("PIM-Error: Unsupported PIM allocation type %d\n", static_cast<int>(allocType));
    return -1;
  }

  if (numCoresReqd > m_device->getNumCores()) {
    std::printf("PIM-Error: Failed to allocate as obj requires %u cores more than available\n", numCoresReqd);
    return -1;
  }

  // create regions
  if (allocType == PIM_ALLOC_V1 || allocType == PIM_ALLOC_H1) {
    for (unsigned i = 0; i < numRegions; ++i) {
      PimCoreId coreId = sortedCoreId[i];
      unsigned numColsToAlloc = (i == numRegions - 1 ? numColsToAllocLast : numCols);
      pimRegion newRegion = findAvailRegionOnCore(coreId, numRowsToAlloc, numColsToAlloc);
      if (!newRegion.isValid()) {
        std::printf("PIM-Error: Failed to allocate object with %u rows on core %d\n", numRowsToAlloc, coreId);
        return -1;
      }
      newObj.addRegion(newRegion);
    }
  }

  // update new object to resource mgr
  m_objMap.insert(std::make_pair(newObj.getObjId(), newObj));
  for (const auto& region : newObj.getRegions()) {
    PimCoreId coreId = region.getCoreId();
    unsigned rowIdx = region.getRowIdx();
    unsigned numAllocRows = region.getNumAllocRows();
    m_coreUsage[coreId].insert(std::make_pair(rowIdx, numAllocRows));
  }

  newObj.print();

  return newObj.getObjId();
}

//! @brief  Alloc a PIM object assiciated to a reference object
//!         For V layout, expect same number of elements, while bits per element may be different
//!         For H layout, expect exact same number of elements and bits per elements
PimObjId
pimResMgr::pimAllocAssociated(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement, PimObjId refId, PimDataType dataType)
{
  // check if ref obj is valid
  if (m_objMap.find(refId) == m_objMap.end()) {
    std::printf("PIM-Error: Invalid ref object ID %d for PIM allocation\n", refId);
    return -1;
  }

  // get regions of the ref obj
  const pimObjInfo& refObj = m_objMap.at(refId);

  // check if the request can be associated with ref
  if (numElements != refObj.getNumElements()) {
    std::printf("PIM-Error: Cannot allocate %u elements associated with ref object ID %d which has %u elements\n",
                numElements, refId, refObj.getNumElements());
    return -1;
  }
  if (allocType == PIM_ALLOC_H1) {
    if (bitsPerElement != refObj.getBitsPerElement()) {
      std::printf("PIM-Error: Cannot allocate elements of %u bits associated with ref object ID %d with %u bits in H1 style\n",
                  bitsPerElement, refId, refObj.getBitsPerElement());
      return -1;
    }
  }

  // allocate regions
  pimObjInfo newObj(m_availObjId, dataType, allocType, numElements, bitsPerElement);
  m_availObjId++;

  for ( const pimRegion& region : refObj.getRegions()) {
    PimCoreId coreId = region.getCoreId();
    unsigned numAllocRows = region.getNumAllocRows();
    unsigned numAllocCols = region.getNumAllocCols();
    if (allocType == PIM_ALLOC_V1) {
      numAllocRows = bitsPerElement;
    }
    pimRegion newRegion = findAvailRegionOnCore(coreId, numAllocRows, numAllocCols);
    if (!newRegion.isValid()) {
      std::printf("PIM-Error: Failed to allocate associated object with %u rows on core %d\n", numAllocRows, coreId);
      return -1;
    }
    newObj.addRegion(newRegion);
  }

  // update new object to resource mgr
  m_objMap.insert(std::make_pair(newObj.getObjId(), newObj));
  for (const auto& region : newObj.getRegions()) {
    PimCoreId coreId = region.getCoreId();
    unsigned rowIdx = region.getRowIdx();
    unsigned numAllocRows = region.getNumAllocRows();
    m_coreUsage[coreId].insert(std::make_pair(rowIdx, numAllocRows));
  }

  newObj.print();

  return newObj.getObjId();
}

//! @brief  Free a PIM object
bool
pimResMgr::pimFree(PimObjId objId)
{
  if (m_objMap.find(objId) == m_objMap.end()) {
    std::printf("PIM-Error: Cannot free non-exist object ID %d\n", objId);
    return false;
  }
  const pimObjInfo& obj = m_objMap.at(objId);
  for (const pimRegion& region : obj.getRegions()) {
    PimCoreId coreId = region.getCoreId();
    unsigned rowIdx = region.getRowIdx();
    unsigned numAllocRows = region.getNumAllocRows();
    m_coreUsage[coreId].erase(std::make_pair(rowIdx, numAllocRows));
  }
  m_objMap.erase(objId);

  return true;
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
  unsigned prevAvail = 0;
  if (m_coreUsage.find(coreId) != m_coreUsage.end()) {
    for (const auto& it : m_coreUsage.at(coreId)) {
      unsigned rowIdx = it.first;
      unsigned numRows = it.second;
      if (rowIdx - prevAvail >= numAllocRows) {
        region.setRowIdx(prevAvail);
        region.setIsValid(true);
        return region;
      }
      prevAvail = rowIdx + numRows;
    }
  }
  if (m_device->getNumRows() - prevAvail >= numAllocRows) {
    region.setRowIdx(prevAvail);
    region.setIsValid(true);
    return region;
  }

  return region;
}

//! @brief  Get number of allocated rows of a specific core
unsigned
pimResMgr::getCoreUsage(PimCoreId coreId) const
{
  if (m_coreUsage.find(coreId) == m_coreUsage.end()) {
    return 0;
  }
  unsigned usage = 0;
  for (const auto& it : m_coreUsage.at(coreId)) {
    usage += it.second;
  }
  return usage;
}

//! @brief  Get a list of core IDs sorted by least usage
std::vector<PimCoreId>
pimResMgr::getCoreIdsSortedByLeastUsage() const
{
  std::vector<std::pair<unsigned, unsigned>> usages;
  for (unsigned coreId = 0; coreId < m_device->getNumCores(); ++coreId) {
    unsigned usage = getCoreUsage(coreId);
    usages.emplace_back(usage, coreId);
  }
  std::sort(usages.begin(), usages.end());
  std::vector<PimCoreId> result;
  for (const auto& it : usages) {
    result.push_back(it.second);
  }
  return result;
}

