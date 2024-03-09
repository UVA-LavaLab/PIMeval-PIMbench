// File: pimCore.cpp
// PIM Functional Simulator - PIM Core
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimCore.h"
#include <random>
#include <iostream>
#include <cstdio>
#include <iomanip>
#include <sstream>


//! @brief  pimCore ctor
pimCore::pimCore(unsigned numRows, unsigned numCols)
  : m_numRows(numRows),
    m_numCols(numCols),
    m_array(numRows, std::vector<bool>(numCols)),
    m_senseAmpRow(numCols),
    m_senseAmpCol(numRows)
{
  // Initialize memory contents with random 0/1
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 1);
  for (unsigned row = 0; row < m_numRows; ++row) {
    for (unsigned col = 0; col < m_numCols; ++col) {
      m_array[row][col] = dist(gen);
    }
  }
}

//! @brief  pimCore cdor
pimCore::~pimCore()
{
}

//! @brief  Read a memory row to SA
bool
pimCore::readRow(unsigned rowIndex)
{
  if (rowIndex >= m_numRows) {
    std::printf("Error: Out-of-boundary subarray row read: index = %u, numRows = %u\n", rowIndex, m_numRows);
    return false;
  }
  m_senseAmpRow = m_array[rowIndex];
  return true;
}

//! @brief  Read a memory column
bool
pimCore::readCol(unsigned colIndex)
{
  if (colIndex >= m_numCols) {
    std::printf("Error: Out-of-boundary subarray column read: index = %u, numCols = %u\n", colIndex, m_numCols);
    return false;
  }
  for (unsigned row = 0; row < m_numRows; ++row) {
    m_senseAmpCol[row] = m_array[row][colIndex];
  }
  return true;
}

//! @brief  Read triple rows. Original contents of the three rows are replaced with majority values.
bool
pimCore::readTripleRows(unsigned rowIndex1, unsigned rowIndex2, unsigned rowIndex3)
{
  if (rowIndex1 >= m_numRows || rowIndex2 >= m_numRows || rowIndex3 >= m_numRows) {
    std::printf("Error: Out-of-boundary subarray triple-row read: indices = (%u, %u, %u), numRows = %u\n", rowIndex1, rowIndex2, rowIndex3, m_numRows);
    return false;
  }
  for (unsigned col = 0; col < m_numCols; ++col) {
    bool val1 = m_array[rowIndex1][col];
    bool val2 = m_array[rowIndex2][col];
    bool val3 = m_array[rowIndex3][col];
    bool maj = (val1 && val2) || (val1 && val3) || (val2 && val3);
    m_array[rowIndex1][col] = maj;
    m_array[rowIndex2][col] = maj;
    m_array[rowIndex3][col] = maj;
    m_senseAmpRow[col] = maj;
  }
  return true;
}

//! @brief  Read triple columns. Original contents of the three columns are replaced with majority values.
bool
pimCore::readTripleCols(unsigned colIndex1, unsigned colIndex2, unsigned colIndex3)
{
  if (colIndex1 >= m_numCols || colIndex2 >= m_numCols || colIndex3 >= m_numCols) {
    std::printf("Error: Out-of-boundary subarray triple-col read: indices = (%u, %u, %u), numCols = %u\n", colIndex1, colIndex2, colIndex3, m_numCols);
    return false;
  }
  for (unsigned row = 0; row < m_numRows; ++row) {
    bool val1 = m_array[colIndex1][row];
    bool val2 = m_array[colIndex2][row];
    bool val3 = m_array[colIndex3][row];
    bool maj = (val1 && val2) || (val1 && val3) || (val2 && val3);
    m_array[colIndex1][row] = maj;
    m_array[colIndex2][row] = maj;
    m_array[colIndex3][row] = maj;
    m_senseAmpCol[row] = maj;
  }
  return true;
}

//! @brief  Write to a memory row
bool
pimCore::writeRow(unsigned rowIndex)
{
  if (rowIndex >= m_numRows) {
    std::printf("Error: Out-of-boundary subarray row write: index = %u, numRows = %u\n", rowIndex, m_numRows);
    return false;
  }
  m_array[rowIndex] = m_senseAmpRow;
  return true;
}

//! @brief  Write to a memory column
bool
pimCore::writeCol(unsigned colIndex)
{
  if (colIndex >= m_numCols) {
    std::printf("Error: Out-of-boundary subarray column write: index = %u, numCols = %u\n", colIndex, m_numCols);
    return false;
  }
  for (unsigned row = 0; row < m_numRows; ++row) {
    m_array[row][colIndex] = m_senseAmpCol[row];
  }
  return true;
}

//! @brief  Set values to row sense amplifiers
bool
pimCore::setSenseAmpRow(const std::vector<bool>& vals)
{
  if (vals.size() != m_numCols) {
    std::printf("Error: Incorrect data size write to row SAs: size = %lu, numCols = %u\n", vals.size(), m_numCols);
    return false;
  }
  m_senseAmpRow = vals;
  return true;
}

//! @brief  Set values to row sense amplifiers
bool
pimCore::setSenseAmpCol(const std::vector<bool>& vals)
{
  if (vals.size() != m_numRows) {
    std::printf("Error: Incorrect data size write to col SAs: size = %lu, numRows = %u\n", vals.size(), m_numRows);
    return false;
  }
  m_senseAmpCol = vals;
  return true;
}

//! @brief  Print out memory subarray contents
void
pimCore::print() const
{
  std::ostringstream oss;
  // header
  oss << "  Row S ";
  for (unsigned col = 0; col < m_array[0].size(); ++col) {
    oss << (col % 8 == 0 ? '+' : '-');
  }
  oss << std::endl;
  for (unsigned row = 0; row < m_array.size(); ++row) {
    // row index
    oss << std::setw(5) << row << ' ';
    // col SA
    oss << m_senseAmpCol[row] << ' ';
    // row contents
    for (unsigned col = 0; col < m_array[0].size(); ++col) {
      oss << m_array[row][col];
    }
    oss << std::endl;
  }
  // footer
  oss << "        ";
  for (unsigned col = 0; col < m_array[0].size(); ++col) {
    oss << (col % 8 == 0 ? '+' : '-');
  }
  oss << std::endl;
  // row SA
  oss << "     SA ";
  for (unsigned col = 0; col < m_array[0].size(); ++col) {
    oss << m_senseAmpRow[col];
  }
  oss << std::endl;
  std::cout << oss.str() << std::endl;
}

