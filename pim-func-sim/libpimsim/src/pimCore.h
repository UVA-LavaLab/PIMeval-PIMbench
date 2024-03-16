// File: pimCore.h
// PIM Functional Simulator - PIM Core
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_PIM_CORE_H
#define LAVA_PIM_CORE_H

#include "libpimsim.h"
#include <vector>
#include <string>
#include <map>


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
  void setBit(unsigned rowIdx, unsigned colIdx, bool val);
  bool getBit(unsigned rowIdx, unsigned colIdx) const;
  void setB32V(unsigned rowIdx, unsigned colIdx, unsigned val);
  unsigned getB32V(unsigned rowIdx, unsigned colIdx) const;
  void setB32H(unsigned rowIdx, unsigned colIdx, unsigned val);
  unsigned getB32H(unsigned rowIdx, unsigned colIdx) const;

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

