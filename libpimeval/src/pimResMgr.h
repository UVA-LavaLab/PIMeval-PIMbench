// File: pimResMgr.h
// PIMeval Simulator - PIM Resource Manager
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_RES_MGR_H
#define LAVA_PIM_RES_MGR_H

#include "libpimeval.h"      // for PimObjId, PimDataType
#include "pimUtils.h"        // for getNumBitsOfDataType, signExt, pimDataTypeEnumToStr, castTypeToBits
#include <vector>            // for vector
#include <unordered_map>     // for unordered_map
#include <set>               // for set
#include <map>               // for map
#include <string>            // for string
#include <memory>            // for unique_ptr
#include <cassert>           // for assert
#include <iostream>          // for cout

class pimDevice;


//! @class  pimRegion
//! @brief  Represent a rectangle region in a PIM core
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

//! @class  pimDataHolder
//! @brief  A container holding raw data vector of a PIM object as a byte array
//! Assumption: Caller gurantees correct range and indices
class pimDataHolder
{
public:
  pimDataHolder(PimDataType dataType, uint64_t numElements)
    : m_dataType(dataType),
      m_numElements(numElements)
  {
    unsigned numBitsOfDataType = pimUtils::getNumBitsOfDataType(m_dataType);
    // Note: Each data element is stored as m_bytesPerElement bytes in this data holder.
    // This aligns with the number of bytes per element in the host void* ptr for memcpy.
    m_bytesPerElement = (numBitsOfDataType + 7) / 8;  // round up, e.g. 1 byte per bool
    m_data.resize(m_numElements * m_bytesPerElement);
  }
  ~pimDataHolder() {}

  // return the number of bytes within a given range
  uint64_t getNumBytes(uint64_t idxBegin, uint64_t idxEnd) const {
    uint64_t numElements = (idxEnd == 0 ? m_numElements : idxEnd - idxBegin);
    return numElements * m_bytesPerElement;
  }

  // copy data of range [idxBegin, idxEnd) from host ptr into holder
  // use full range if idxEnd is default 0
  bool copyFromHost(void* src, uint64_t idxBegin = 0, uint64_t idxEnd = 0) {
    uint64_t byteIndex = idxBegin * m_bytesPerElement;
    uint64_t numBytes = getNumBytes(idxBegin, idxEnd);
    std::memcpy(m_data.data() + byteIndex, src, numBytes);
    return true;
  }

  // copy data of range [idxBegin, idxEnd) from holder to host ptr
  // use full range if idxEnd is default 0
  bool copyToHost(void* dest, uint64_t idxBegin = 0, uint64_t idxEnd = 0) const {
    uint64_t byteIndex = idxBegin * m_bytesPerElement;
    uint64_t numBytes = getNumBytes(idxBegin, idxEnd);
    std::memcpy(dest, m_data.data() + byteIndex, numBytes);
    return true;
  }

  // copy data of range [idxBegin, idxEnd) from this holder to another holder
  // use full range if idxEnd is default 0
  bool copyToObj(pimDataHolder& dest, uint64_t idxBegin = 0, uint64_t idxEnd = 0) const {
    uint64_t byteIndex = idxBegin * m_bytesPerElement;
    uint64_t numBytes = getNumBytes(idxBegin, idxEnd);
    std::memcpy(dest.m_data.data() + byteIndex, m_data.data() + byteIndex, numBytes);
    return true;
  }

  // set an element at index from bit representation
  bool setElementBits(uint64_t index, uint64_t bits) {
    uint64_t byteIndex = index * m_bytesPerElement;
    std::memcpy(m_data.data() + byteIndex, &bits, m_bytesPerElement);
    return true;
  }

  // get bit representation of an element at index
  bool getElementBits(uint64_t index, uint64_t &bits) const {
    bits = 0;
    uint64_t byteIndex = index * m_bytesPerElement;
    std::memcpy(&bits, m_data.data() + byteIndex, m_bytesPerElement);
    bits = pimUtils::signExt(bits, m_dataType);
    return true;
  }

  // print all bytes for debugging
  void print() const {
    std::cout << "PIM obj data holder: data-type = " << pimUtils::pimDataTypeEnumToStr(m_dataType)
              << ", num-elements = " << m_numElements
              << ", bytes-per-element = " << m_bytesPerElement << std::endl;
    for (size_t i = 0; i < m_data.size(); ++i) {
      std::printf(" %02x", m_data[i]);
      if ((i + 1) % 64 == 0) { std::printf("\n"); }
    }
    std::printf("\n");
  }

private:
  std::vector<uint8_t> m_data;
  PimDataType m_dataType;
  uint64_t m_numElements;
  unsigned m_bytesPerElement;
};

//! @class  pimObjInfo
//! @brief  Meta data of a PIM object which includes
//!         - PIM object ID
//!         - One or more rectangle regions allocated in one or more PIM cores
//!         - Allocation type which specifies how data is stored in a region
class pimObjInfo
{
public:
  pimObjInfo(PimObjId objId, PimDataType dataType, PimAllocEnum allocType, uint64_t numElements, unsigned bitsPerElement, pimDevice* device)
    : m_objId(objId),
      m_assocObjId(objId),
      m_dataType(dataType),
      m_allocType(allocType),
      m_data(dataType, numElements),
      m_numElements(numElements),
      m_bitsPerElement(bitsPerElement),
      m_device(device)
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
  pimDevice* getDevice() { return m_device; }
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

  // Note: Below functions are wraper APIs to access PIM object data holder
  // For regular PIM objects:
  // - Support host-to-device, device-to-host, and device-to-device copying
  // - Use bit representation to set or get an element at specific element index
  // - Support ranges in [idxBegin, idxEnd). Use full range if idxEnd is 0
  // For reference PIM objects:
  // - A ref object directly access the data holder of the ref-to object
  // - Dual-contact ref negates all bits during operations
  void copyFromHost(void* src, uint64_t idxBegin = 0, uint64_t idxEnd = 0);
  void copyToHost(void* dest, uint64_t idxBegin = 0, uint64_t idxEnd = 0) const;
  void copyToObj(pimObjInfo& destObj, uint64_t idxBegin = 0, uint64_t idxEnd = 0) const;
  void setElementBits(uint64_t index, uint64_t bits);
  uint64_t getElementBits(uint64_t index) const;
  template <typename T> void setElement(uint64_t index, T val) {
    setElementBits(index, pimUtils::castTypeToBits(val));
  }

  // Note: Below two functions are for supporting mixed functional and micro-ops level simulation.
  // Functional simulation purely uses this PIM data holder for simulation speed,
  // while micro-ops level simulation uses simulated 2D memory arrays.
  // When a functional API is called during micro-ops level simulation, call below two functions
  // to sync the data between this PIM data holder and simulated memory arrays.
  void syncFromSimulatedMem();
  void syncToSimulatedMem() const;

private:
  PimObjId m_objId = -1;
  PimObjId m_assocObjId = -1;
  PimObjId m_refObjId = -1;
  PimDataType m_dataType;
  PimAllocEnum m_allocType;
  pimDataHolder m_data;
  uint64_t m_numElements = 0;
  unsigned m_bitsPerElement = 0;
  std::vector<pimRegion> m_regions;  // a list of core ID and regions
  unsigned m_maxNumRegionsPerCore = 0;
  unsigned m_numCoresUsed = 0;
  unsigned m_maxElementsPerRegion = 0;
  unsigned m_numColsPerElem = 0; // number of cols per element
  bool m_isDualContactRef = false;
  pimDevice* m_device = nullptr; // for accessing simulated memory
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
  pimObjInfo& getObjInfo(PimObjId objId) { return m_objMap.at(objId); }

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

