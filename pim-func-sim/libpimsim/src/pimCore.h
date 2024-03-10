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
  const std::vector<bool>& getSenseAmpRow() const { return m_senseAmpRow; }
  bool setSenseAmpRow(const std::vector<bool>& vals);

  // Column-based operations
  bool readCol(unsigned colIndex);
  bool readTripleCols(unsigned colIndex1, unsigned colIndex2, unsigned colIndex3);
  bool writeCol(unsigned colIndex);
  const std::vector<bool>& getSenseAmpCol() const { return m_senseAmpCol; }
  bool setSenseAmpCol(const std::vector<bool>& vals);

  // Utilities
  bool declareRowReg(const std::string& name);
  bool declareColReg(const std::string& name);
  void print() const;

  void setBit(unsigned rowIdx, unsigned colIdx, bool val) { m_array[rowIdx][colIdx] = val; }
  bool getBit(unsigned rowIdx, unsigned colIdx) const { return m_array[rowIdx][colIdx]; }

private:
  PimCoreId m_coreId;
  unsigned m_numRows;
  unsigned m_numCols;

  std::vector<std::vector<bool>> m_array;
  std::vector<bool> m_senseAmpRow;
  std::vector<bool> m_senseAmpCol;

  std::map<std::string, std::vector<bool>> m_rowRegs;
  std::map<std::string, std::vector<bool>> m_colRegs;
};

#endif

