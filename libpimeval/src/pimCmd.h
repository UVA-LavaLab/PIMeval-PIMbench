// File: pimCmd.h
// PIMeval Simulator - PIM Commands
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_CMD_H
#define LAVA_PIM_CMD_H

#include "libpimeval.h"      // for PimDataType, PimObjId
#include "pimResMgr.h"       // for pimResMgr, pimObjInfo
#include "pimCore.h"         // for pimCore
#include "pimUtils.h"        // for pimDataTypeEnumToStr, threadWorker
#include <vector>            // for vector
#include <string>            // for string
#include <limits>            // for numeric_limits
#include <cassert>           // for assert
#include <bitset>            // for bitset

class pimDevice;


enum class PimCmdEnum {
  NOOP = 0,
  COPY_H2D,
  COPY_D2H,
  COPY_D2D,
  // Functional 1-operand
  ABS,
  POPCOUNT,
  SHIFT_BITS_R,
  SHIFT_BITS_L,
  ADD_SCALAR,
  SUB_SCALAR,
  MUL_SCALAR,
  DIV_SCALAR,
  AND_SCALAR,
  OR_SCALAR,
  XOR_SCALAR,
  XNOR_SCALAR,
  GT_SCALAR,
  LT_SCALAR,
  EQ_SCALAR,
  MIN_SCALAR,
  MAX_SCALAR,
  // Functional 2-operand
  ADD,
  SUB,
  MUL,
  SCALED_ADD,
  DIV,
  AND,
  OR,
  XOR,
  XNOR,
  GT,
  LT,
  EQ,
  MIN,
  MAX,
  // Functional special
  REDSUM,
  REDSUM_RANGE,
  BROADCAST,
  ROTATE_ELEM_R,
  ROTATE_ELEM_L,
  SHIFT_ELEM_R,
  SHIFT_ELEM_L,

  // BitSIMD v-layout commands
  ROW_R,
  ROW_W,
  RREG_MOV,
  RREG_SET,
  RREG_NOT,
  RREG_AND,
  RREG_OR,
  RREG_NAND,
  RREG_NOR,
  RREG_XOR,
  RREG_XNOR,
  RREG_MAJ,
  RREG_SEL,
  RREG_ROTATE_R,
  RREG_ROTATE_L,
  // SIMDRAM
  ROW_AP,
  ROW_AAP,
};


//! @class  pimCmd
//! @brief  Pim command base class
class pimCmd
{
public:
  pimCmd(PimCmdEnum cmdType) : m_cmdType(cmdType) {}
  virtual ~pimCmd() {}

  void setDevice(pimDevice* device) { m_device = device; }
  virtual bool execute() = 0;

  std::string getName() const {
    return getName(m_cmdType, "");
  }
  std::string getName(PimDataType dataType, bool isVLayout) const {
    std::string suffix = "." + pimUtils::pimDataTypeEnumToStr(dataType);
    suffix += isVLayout ? ".v" : ".h";
    return getName(m_cmdType, suffix);
  }
  static std::string getName(PimCmdEnum cmdType, const std::string& suffix);

protected:
  bool isValidObjId(pimResMgr* resMgr, PimObjId objId) const;
  bool isAssociated(const pimObjInfo& obj1, const pimObjInfo& obj2) const;
  bool isCompatibleType(const pimObjInfo& obj1, const pimObjInfo& obj2) const;
  bool isConvertibleType(const pimObjInfo& src, const pimObjInfo& dest) const;

  unsigned getNumElementsInRegion(const pimRegion& region, unsigned bitsPerElement) const;

  virtual bool sanityCheck() const { return false; }
  virtual bool computeRegion(unsigned index) { return false; }
  virtual bool updateStats() const { return false; }
  bool computeAllRegions(unsigned numRegions);

  //! @brief  Utility: Get bits of an element from a region. The bits are stored as uint64_t without sign extension
  inline uint64_t getBits(const pimCore& core, bool isVLayout, unsigned rowLoc, unsigned colLoc, unsigned numBits) const
  {
    return isVLayout ? core.getBitsV(rowLoc, colLoc, numBits) : core.getBitsH(rowLoc, colLoc, numBits);
  }

  //! @brief  Utility: Set bits of an element to a region
  inline void setBits(pimCore& core, bool isVLayout, unsigned rowLoc, unsigned colLoc, uint64_t bits, unsigned numBits) const
  {
    if (isVLayout) {
      core.setBitsV(rowLoc, colLoc, bits, numBits);
    } else {
      core.setBitsH(rowLoc, colLoc, bits, numBits);
    }
  }

  PimCmdEnum m_cmdType;
  pimDevice* m_device = nullptr;

  //! @class  pimCmd::regionWorker
  //! @brief  Thread worker to process regions in parallel
  class regionWorker : public pimUtils::threadWorker {
  public:
    regionWorker(pimCmd* cmd, unsigned regionIdx) : m_cmd(cmd), m_regionIdx(regionIdx) {}
    virtual ~regionWorker() {}
    virtual void execute() {
      m_cmd->computeRegion(m_regionIdx);
    }
  private:
    pimCmd* m_cmd = nullptr;
    unsigned m_regionIdx = 0;
  };
};

//! @class  pimCmdDataTransfer
//! @brief  Data transfer. Not tracked as a regular Pim CMD
class pimCmdCopy : public pimCmd
{
public:
  pimCmdCopy(PimCmdEnum cmdType, PimCopyEnum copyType, void* src, PimObjId dest, uint64_t idxBegin = 0, uint64_t idxEnd = 0)
    : pimCmd(PimCmdEnum::COPY_H2D), m_copyType(copyType), m_ptr(src), m_dest(dest), m_idxBegin(idxBegin), m_idxEnd(idxEnd), m_copyFullRange(idxEnd == 0ULL) {}
  pimCmdCopy(PimCmdEnum cmdType, PimCopyEnum copyType, PimObjId src, void* dest, uint64_t idxBegin = 0, uint64_t idxEnd = 0)
    : pimCmd(PimCmdEnum::COPY_D2H), m_copyType(copyType), m_ptr(dest), m_src(src), m_idxBegin(idxBegin), m_idxEnd(idxEnd), m_copyFullRange(idxEnd == 0ULL) {}
  pimCmdCopy(PimCmdEnum cmdType, PimCopyEnum copyType, PimObjId src, PimObjId dest, uint64_t idxBegin = 0, uint64_t idxEnd = 0)
    : pimCmd(PimCmdEnum::COPY_D2D), m_copyType(copyType), m_src(src), m_dest(dest), m_idxBegin(idxBegin), m_idxEnd(idxEnd), m_copyFullRange(idxEnd == 0ULL) {}

  virtual ~pimCmdCopy() {}
  virtual bool execute() override;
  virtual bool sanityCheck() const override;
  virtual bool updateStats() const override;
protected:
  PimCopyEnum m_copyType;
  void* m_ptr = nullptr;
  PimObjId m_src = -1;
  PimObjId m_dest = -1;
  uint64_t m_idxBegin = 0;
  uint64_t m_idxEnd = 0; 
  bool m_copyFullRange = false;
};

//! @class  pimCmdFunc1
//! @brief  Pim CMD: Functional 1-operand
class pimCmdFunc1 : public pimCmd
{
public:
  pimCmdFunc1(PimCmdEnum cmdType, PimObjId src, PimObjId dest, uint64_t scalarValue = 0)
    : pimCmd(cmdType), m_src(src), m_dest(dest), m_scalarValue(scalarValue) {}
  virtual ~pimCmdFunc1() {}
  virtual bool execute() override;
  virtual bool sanityCheck() const override;
  virtual bool computeRegion(unsigned index) override;
  virtual bool updateStats() const override;
protected:
  PimObjId m_src;
  PimObjId m_dest;
  uint64_t m_scalarValue;
private:
  template<typename T>
  inline bool computeResult(T operand, PimCmdEnum cmdType, T scalarValue, T& result, int bitsPerElementSrc) {
    result = operand;
    switch (cmdType) {
    case PimCmdEnum::ADD_SCALAR: result += scalarValue; break;
    case PimCmdEnum::SUB_SCALAR: result -= scalarValue; break;
    case PimCmdEnum::MUL_SCALAR: result *= scalarValue; break;
    case PimCmdEnum::DIV_SCALAR:
        if (scalarValue == 0) {
            std::printf("PIM-Error: Division by zero\n");
            return false;
        }
        result /= scalarValue;
        break;
    case PimCmdEnum::AND_SCALAR: result &= scalarValue; break;
    case PimCmdEnum::OR_SCALAR: result |= scalarValue; break;
    case PimCmdEnum::XOR_SCALAR: result ^= scalarValue; break;
    case PimCmdEnum::XNOR_SCALAR: result = ~(operand ^ scalarValue); break;
    case PimCmdEnum::GT_SCALAR: result = (operand > scalarValue) ? 1 : 0; break;
    case PimCmdEnum::LT_SCALAR: result = (operand < scalarValue) ? 1 : 0; break;
    case PimCmdEnum::EQ_SCALAR: result = (operand == scalarValue) ? 1 : 0; break;
    case PimCmdEnum::MIN_SCALAR: result = std::min(operand, scalarValue); break;
    case PimCmdEnum::MAX_SCALAR: result = std::max(operand, scalarValue); break;
    case PimCmdEnum::POPCOUNT:
        switch (bitsPerElementSrc) {
        case 8: result = std::bitset<8>(operand).count(); break;
        case 16: result = std::bitset<16>(operand).count(); break;
        case 32: result = std::bitset<32>(operand).count(); break;
        case 64: result = std::bitset<64>(operand).count(); break;
        default:
            std::printf("PIM-Error: Unsupported bits per element %u\n", bitsPerElementSrc);
            return false;
        }
        break;
    case PimCmdEnum::SHIFT_BITS_R: result >>= static_cast<uint64_t>(scalarValue); break;
    case PimCmdEnum::SHIFT_BITS_L: result <<= static_cast<uint64_t>(scalarValue); break;
    case PimCmdEnum::ABS:
    {
        if (std::is_signed<T>::value) {
          result = (operand < 0) ? -operand : operand;
        } else {
          result = operand;
        }
        break;
    }
    default:
        std::printf("PIM-Error: Unexpected cmd type %d\n", static_cast<int>(cmdType));
        assert(0);
    }
    return true;
  }

  template<typename T>
  inline bool computeResultFP(T operand, PimCmdEnum cmdType, T scalerValue, T& result, int bitsPerElementSrc) {
    result = operand;
    switch (cmdType) {
    case PimCmdEnum::ADD_SCALAR: result += scalerValue; break;
    case PimCmdEnum::SUB_SCALAR: result -= scalerValue; break;
    case PimCmdEnum::MUL_SCALAR: result *= scalerValue; break;
    case PimCmdEnum::DIV_SCALAR:
        if (scalerValue == 0) {
            std::printf("PIM-Error: Division by zero\n");
            return false;
        }
        result /= scalerValue;
        break;
    case PimCmdEnum::GT_SCALAR: result = (operand > scalerValue) ? 1 : 0; break;
    case PimCmdEnum::LT_SCALAR: result = (operand < scalerValue) ? 1 : 0; break;
    case PimCmdEnum::EQ_SCALAR: result = (operand == scalerValue) ? 1 : 0; break;
    case PimCmdEnum::MIN_SCALAR: result = std::min(operand, scalerValue); break;
    case PimCmdEnum::MAX_SCALAR: result = std::max(operand, scalerValue); break;
    case PimCmdEnum::ABS:
    {
        if (std::is_signed<T>::value) {
          result = (operand < 0) ? -operand : operand;
        } else {
          result = operand;
        }
        break;
    }
    case PimCmdEnum::AND_SCALAR:
    case PimCmdEnum::OR_SCALAR:
    case PimCmdEnum::XOR_SCALAR:
    case PimCmdEnum::XNOR_SCALAR:
    case PimCmdEnum::POPCOUNT:
    case PimCmdEnum::SHIFT_BITS_R:
    case PimCmdEnum::SHIFT_BITS_L:
        std::printf("PIM-Error: Cannot perform bitwise operation on floating point values.\n");
        return false;
    default:
        std::printf("PIM-Error: Unexpected cmd type %d\n", static_cast<int>(cmdType));
        assert(0);
    }
    return true;
  }
};

//! @class  pimCmdFunc2
//! @brief  Pim CMD: Functional 2-operand
class pimCmdFunc2 : public pimCmd
{
public:
  pimCmdFunc2(PimCmdEnum cmdType, PimObjId src1, PimObjId src2, PimObjId dest)
    : pimCmd(cmdType), m_src1(src1), m_src2(src2), m_dest(dest) {}
  pimCmdFunc2(PimCmdEnum cmdType, PimObjId src1, PimObjId src2, PimObjId dest, uint64_t scalarValue)
    : pimCmd(cmdType), m_src1(src1), m_src2(src2), m_dest(dest), m_scalarValue(scalarValue) {}
  virtual ~pimCmdFunc2() {}
  virtual bool execute() override;
  virtual bool sanityCheck() const override;
  virtual bool computeRegion(unsigned index) override;
  virtual bool updateStats() const override;
protected:
  PimObjId m_src1;
  PimObjId m_src2;
  PimObjId m_dest;
  uint64_t m_scalarValue;
};

//! @class  pimCmdedSum
//! @brief  Pim CMD: RedSum non-ranged/ranged
template <typename T> class pimCmdRedSum : public pimCmd
{
public:
  pimCmdRedSum(PimCmdEnum cmdType, PimObjId src, T* result)
    : pimCmd(cmdType), m_src(src), m_result(result)
  {
    assert(cmdType == PimCmdEnum::REDSUM);
  }
  pimCmdRedSum(PimCmdEnum cmdType, PimObjId src, T* result, uint64_t idxBegin, uint64_t idxEnd)
    : pimCmd(cmdType), m_src(src), m_result(result), m_idxBegin(idxBegin), m_idxEnd(idxEnd)
  {
    assert(cmdType == PimCmdEnum::REDSUM_RANGE);
  }
  virtual ~pimCmdRedSum() {}
  virtual bool execute() override;
  virtual bool sanityCheck() const override;
  virtual bool computeRegion(unsigned index) override;
  virtual bool updateStats() const override;
protected:
  PimObjId m_src;
  T* m_result;
  std::vector<T> m_regionSum;
  uint64_t m_idxBegin = 0;
  uint64_t m_idxEnd = std::numeric_limits<uint64_t>::max();
};

//! @class  pimCmdBroadcast
//! @brief  Pim CMD: Broadcast a value to all elements
class pimCmdBroadcast : public pimCmd
{
public:
  pimCmdBroadcast(PimCmdEnum cmdType, PimObjId dest, uint64_t signExtBits)
    : pimCmd(cmdType), m_dest(dest), m_signExtBits(signExtBits)
  {
    assert(cmdType == PimCmdEnum::BROADCAST);
  }
  virtual ~pimCmdBroadcast() {}
  virtual bool execute() override;
  virtual bool sanityCheck() const override;
  virtual bool computeRegion(unsigned index) override;
  virtual bool updateStats() const override;
protected:
  PimObjId m_dest;
  uint64_t m_signExtBits;
};

//! @class  pimCmdRotate
//! @brief  Pim CMD: rotate/shift elements right/left
class pimCmdRotate : public pimCmd
{
public:
  pimCmdRotate(PimCmdEnum cmdType, PimObjId src)
    : pimCmd(cmdType), m_src(src)
  {
    assert(cmdType == PimCmdEnum::ROTATE_ELEM_R || cmdType == PimCmdEnum::ROTATE_ELEM_L ||
           cmdType == PimCmdEnum::SHIFT_ELEM_R || cmdType == PimCmdEnum::SHIFT_ELEM_L);
  }
  virtual ~pimCmdRotate() {}
  virtual bool execute() override;
  virtual bool sanityCheck() const override;
  virtual bool computeRegion(unsigned index) override;
  virtual bool updateStats() const override;
protected:
  PimObjId m_src;
  std::vector<uint64_t> m_regionBoundary;
};

//! @class  pimCmdReadRowToSa
//! @brief  Pim CMD: BitSIMD-V: Read a row to SA
class pimCmdReadRowToSa : public pimCmd
{
public:
  pimCmdReadRowToSa(PimCmdEnum cmdType, PimObjId objId, unsigned ofst)
    : pimCmd(cmdType), m_objId(objId), m_ofst(ofst) {}
  virtual ~pimCmdReadRowToSa() {}
  virtual bool execute() override;
protected:
  PimObjId m_objId;
  unsigned m_ofst;
};

//! @class  pimCmdWriteSaToRow
//! @brief  Pim CMD: BitSIMD-V: Write SA to a row
class pimCmdWriteSaToRow : public pimCmd
{
public:
  pimCmdWriteSaToRow(PimCmdEnum cmdType, PimObjId objId, unsigned ofst)
    : pimCmd(cmdType), m_objId(objId), m_ofst(ofst) {}
  virtual ~pimCmdWriteSaToRow() {}
  virtual bool execute() override;
protected:
  PimObjId m_objId;
  unsigned m_ofst;
};

//! @class  pimCmdRRegOp : public pimCmd
//! @brief  Pim CMD: BitSIMD-V: Row reg operations
class pimCmdRRegOp : public pimCmd
{
public:
  pimCmdRRegOp(PimCmdEnum cmdType, PimObjId objId, PimRowReg dest, bool val)
    : pimCmd(cmdType), m_objId(objId), m_dest(dest), m_val(val)
  {
    assert(cmdType == PimCmdEnum::RREG_SET);
  }
  pimCmdRRegOp(PimCmdEnum cmdType, PimObjId objId, PimRowReg dest, PimRowReg src1)
    : pimCmd(cmdType), m_objId(objId), m_dest(dest), m_src1(src1)
  {
    assert(cmdType == PimCmdEnum::RREG_MOV || cmdType == PimCmdEnum::RREG_NOT);
  }
  pimCmdRRegOp(PimCmdEnum cmdType, PimObjId objId, PimRowReg dest, PimRowReg src1, PimRowReg src2)
    : pimCmd(cmdType), m_objId(objId), m_dest(dest), m_src1(src1), m_src2(src2)
  {
  }
  pimCmdRRegOp(PimCmdEnum cmdType, PimObjId objId, PimRowReg dest, PimRowReg src1, PimRowReg src2, PimRowReg src3)
    : pimCmd(cmdType), m_objId(objId), m_dest(dest), m_src1(src1), m_src2(src2), m_src3(src3)
  {
    assert(cmdType == PimCmdEnum::RREG_MAJ || cmdType == PimCmdEnum::RREG_SEL);
  }
  virtual ~pimCmdRRegOp() {}
  virtual bool execute() override;
protected:
  PimObjId m_objId;
  PimRowReg m_dest;
  bool m_val = 0;
  PimRowReg m_src1 = PIM_RREG_NONE;
  PimRowReg m_src2 = PIM_RREG_NONE;
  PimRowReg m_src3 = PIM_RREG_NONE;
};

//! @class  pimCmdRRegRotate
//! @brief  Pim CMD: BitSIMD-V: row reg rotate right by one step
class pimCmdRRegRotate : public pimCmd
{
public:
  pimCmdRRegRotate(PimCmdEnum cmdType, PimObjId objId, PimRowReg dest)
    : pimCmd(cmdType), m_objId(objId), m_dest(dest) {}
  virtual ~pimCmdRRegRotate() {}
  virtual bool execute() override;
protected:
  PimObjId m_objId;
  PimRowReg m_dest;
};

//! @class  pimCmdAnalogAAP
//! @brief  Pim CMD: SIMDRAM: Analog based multi-row AP (activate-precharge) or AAP (activate-activate-precharge)
class pimCmdAnalogAAP : public pimCmd
{
public:
  pimCmdAnalogAAP(PimCmdEnum cmdType,
                  const std::vector<std::pair<PimObjId, unsigned>>& srcRows,
                  const std::vector<std::pair<PimObjId, unsigned>>& destRows = {})
    : pimCmd(cmdType), m_srcRows(srcRows), m_destRows(destRows)
  {
    assert(cmdType == PimCmdEnum::ROW_AP || cmdType == PimCmdEnum::ROW_AAP);
  }
  virtual ~pimCmdAnalogAAP() {}
  virtual bool execute() override;
protected:
  void printDebugInfo() const;
  std::vector<std::pair<PimObjId, unsigned>> m_srcRows;
  std::vector<std::pair<PimObjId, unsigned>> m_destRows;
};

#endif

