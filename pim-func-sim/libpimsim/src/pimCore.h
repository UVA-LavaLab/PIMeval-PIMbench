// File: pimCore.h
// PIM Functional Simulator - PIM Core
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_CORE_H
#define LAVA_PIM_CORE_H

#include "libpimsim.h"
#include <vector>
#include <string>
#include <map>
#include <cassert>


//! @class  pimCore
//! @brief  A PIM core which performs computation on a 2D memory subarray
class pimCore
{
public:
  pimCore(unsigned numRows, unsigned numCols);
  ~pimCore();

  // ID
  void setCoreId(int id) { m_coreId = id; }
  PimCoreId getCoreId() const { return m_coreId; }

  // Row-based operations
  bool readRow(unsigned rowIndex);
  bool readTripleRows(unsigned rowIndex1, unsigned rowIndex2, unsigned rowIndex3);
  bool writeRow(unsigned rowIndex);
  std::vector<bool>& getSenseAmpRow() { return m_rowRegs[PIM_RREG_SA]; }
  bool setSenseAmpRow(const std::vector<bool>& vals);

  // Column-based operations
  bool readCol(unsigned colIndex);
  bool readTripleCols(unsigned colIndex1, unsigned colIndex2, unsigned colIndex3);
  bool writeCol(unsigned colIndex);
  std::vector<bool>& getSenseAmpCol() { return m_senseAmpCol; }
  bool setSenseAmpCol(const std::vector<bool>& vals);

  // Reg access
  std::vector<bool>& getRowReg(PimRowReg reg) { return m_rowRegs[reg]; }

  // Utilities
  bool declareRowReg(PimRowReg reg);
  bool declareColReg(const std::string& name);
  void print() const;

  // Directly manipulate bits for functional implementation
  //! @brief  Directly set a bit for functional simulation
  inline void setBit(unsigned rowIdx, unsigned colIdx, bool val) {
    assert(rowIdx < m_numRows && colIdx < m_numCols);
    m_array[rowIdx][colIdx] = val;
  }
  //! @brief  Directly get a bit for functional simulation
  inline bool getBit(unsigned rowIdx, unsigned colIdx) const {
    assert(rowIdx < m_numRows && colIdx < m_numCols);
    return m_array[rowIdx][colIdx];
  }
  //! @brief  Directly set 32 bits for V-layout functional simulation
  inline void setB32V(unsigned rowIdx, unsigned colIdx, unsigned val) {
    assert(rowIdx + 31 < m_numRows && colIdx < m_numCols);
    for (int i = 0; i < 32; ++i) {
      bool bitVal = val & 1;
      setBit(rowIdx + i, colIdx, bitVal);
      val = val >> 1;
    }
  }
  //! @brief  Directly get 32 bits for V-layout functional simulation
  inline unsigned getB32V(unsigned rowIdx, unsigned colIdx) const {
    assert(rowIdx + 31 < m_numRows && colIdx < m_numCols);
    unsigned val = 0;
    for (int i = 31; i >= 0; --i) {
      bool bitVal = getBit(rowIdx + i, colIdx);
      val = (val << 1) | bitVal;
    }
    return val;
  }
  //! @brief  Directly set 32 bits for H-layout functional simulation
  inline void setB32H(unsigned rowIdx, unsigned colIdx, unsigned val) {
    assert(rowIdx < m_numRows && colIdx + 31 < m_numCols);
    for (int i = 0; i < 32; ++i) {
      bool bitVal = val & 1;
      setBit(rowIdx, colIdx + i, bitVal);
      val = val >> 1;
    }
  }
  //! @brief  Directly get 32 bits for H-layout functional simulation
  inline unsigned getB32H(unsigned rowIdx, unsigned colIdx) const {
    assert(rowIdx < m_numRows && colIdx + 31 < m_numCols);
    unsigned val = 0;
    for (int i = 31; i >= 0; --i) {
      bool bitVal = getBit(rowIdx, colIdx + i);
      val = (val << 1) | bitVal;
    }
    return val;
  }

private:
  PimCoreId m_coreId;
  unsigned m_numRows;
  unsigned m_numCols;

  std::vector<std::vector<bool>> m_array;
  std::vector<bool> m_senseAmpCol;

  std::map<PimRowReg, std::vector<bool>> m_rowRegs;
  std::map<std::string, std::vector<bool>> m_colRegs;
};

#endif

