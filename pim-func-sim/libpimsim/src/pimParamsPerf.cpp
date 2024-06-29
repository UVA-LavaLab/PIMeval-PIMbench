// File: pimParamsPerf.cc
// PIM Functional Simulator - Performance parameters
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "pimParamsPerf.h"
#include "pimSim.h"
#include "pimCmd.h"
#include <cstdio>
#include <unordered_map>
#include <tuple>


//! @brief  BitSIMD performance table (Tuple: #R, #W, #L)
const std::unordered_map<PimDeviceEnum, std::unordered_map<PimDataType,
    std::unordered_map<PimCmdEnum, std::tuple<unsigned, unsigned, unsigned>>>>
pimParamsPerf::s_bitsimdPerfTable = {
  { PIM_DEVICE_BITSIMD_V, {
    { PIM_INT8, {
      { PimCmdEnum::ABS,          {    9,    8,   34 } },
      //{ PimCmdEnum::POPCOUNT,     {    0,    0,    0 } },
      { PimCmdEnum::ADD,          {   16,    8,   25 } },
      { PimCmdEnum::SUB,          {   16,    8,   25 } },
      { PimCmdEnum::MUL,          {   72,   36,  136 } },
      { PimCmdEnum::DIV,          {  196,  137,  336 } },
      { PimCmdEnum::AND,          {   16,    8,   16 } },
      { PimCmdEnum::OR,           {   16,    8,   16 } },
      { PimCmdEnum::XOR,          {   16,    8,   16 } },
      { PimCmdEnum::XNOR,         {   16,    8,   24 } },
      { PimCmdEnum::GT,           {   16,    8,   26 } },
      { PimCmdEnum::LT,           {   16,    8,   26 } },
      { PimCmdEnum::EQ,           {   16,    8,   27 } },
      { PimCmdEnum::MIN,          {   32,    8,   41 } },
      { PimCmdEnum::MAX,          {   32,    8,   41 } },
    }},
    { PIM_INT16, {
      { PimCmdEnum::ABS,          {   17,   16,   66 } },
      //{ PimCmdEnum::POPCOUNT,     {    0,    0,    0 } },
      { PimCmdEnum::ADD,          {   32,   16,   49 } },
      { PimCmdEnum::SUB,          {   32,   16,   49 } },
      { PimCmdEnum::MUL,          {  272,  136,  528 } },
      { PimCmdEnum::DIV,          {  772,  469, 1176 } },
      { PimCmdEnum::AND,          {   32,   16,   32 } },
      { PimCmdEnum::OR,           {   32,   16,   32 } },
      { PimCmdEnum::XOR,          {   32,   16,   32 } },
      { PimCmdEnum::XNOR,         {   32,   16,   48 } },
      { PimCmdEnum::GT,           {   32,   16,   50 } },
      { PimCmdEnum::LT,           {   32,   16,   50 } },
      { PimCmdEnum::EQ,           {   32,   16,   51 } },
      { PimCmdEnum::MIN,          {   64,   16,   81 } },
      { PimCmdEnum::MAX,          {   64,   16,   81 } },
    }},
    { PIM_INT32, {
      { PimCmdEnum::ABS,          {   33,   32,  130 } },
      { PimCmdEnum::POPCOUNT,     {  114,  114,  218 } },
      { PimCmdEnum::ADD,          {   64,   32,   97 } },
      { PimCmdEnum::SUB,          {   64,   32,   97 } },
      { PimCmdEnum::MUL,          { 1056,  528, 2080 } },
      { PimCmdEnum::DIV,          { 3076, 1709, 4392 } },
      { PimCmdEnum::AND,          {   64,   32,   64 } },
      { PimCmdEnum::OR,           {   64,   32,   64 } },
      { PimCmdEnum::XOR,          {   64,   32,   64 } },
      { PimCmdEnum::XNOR,         {   64,   32,   96 } },
      { PimCmdEnum::GT,           {   64,   32,   98 } },
      { PimCmdEnum::LT,           {   64,   32,   98 } },
      { PimCmdEnum::EQ,           {   64,   32,   99 } },
      { PimCmdEnum::MIN,          {  128,   32,  161 } },
      { PimCmdEnum::MAX,          {  128,   32,  161 } },
      // TODO: Scalers need to be updated with proper numbers. They are currently considering same #write & #logic as the non-scaler ones. But #read has been divided by 2
      //{ PimCmdEnum::ADD_SCALAR,   {   32,   33,  161 } },
      //{ PimCmdEnum::SUB_SCALAR,   {   32,   33,  161 } },
      //{ PimCmdEnum::MUL_SCALAR,   {  970, 1095, 3606 } },
      //{ PimCmdEnum::DIV_SCALAR,   { 1584, 1727, 4257 } },
      //{ PimCmdEnum::AND_SCALAR,   {   32,   32,   64 } },
      //{ PimCmdEnum::OR_SCALAR,    {   32,   32,   64 } },
      //{ PimCmdEnum::XOR_SCALAR,   {   32,   32,   64 } },
      //{ PimCmdEnum::XNOR_SCALAR,  {   32,   32,   64 } },
      //{ PimCmdEnum::GT_SCALAR,    {   32,   32,   66 } },
      //{ PimCmdEnum::LT_SCALAR,    {   32,   32,   66 } },
      //{ PimCmdEnum::EQ_SCALAR,    {   32,   32,   66 } },
      //{ PimCmdEnum::MIN_SCALAR,   {   82,   67,  258 } },
      //{ PimCmdEnum::MAX_SCALAR,   {   82,   67,  258 } },
    }},
    { PIM_INT64, {
      { PimCmdEnum::ABS,          {   65,   64,  258 } },
      //{ PimCmdEnum::POPCOUNT,     {    0,    0,    0 } },
      { PimCmdEnum::ADD,          {  128,   64,  193 } },
      { PimCmdEnum::SUB,          {  128,   64,  193 } },
      //{ PimCmdEnum::MUL,          {    0,    0,    0 } },
      //{ PimCmdEnum::DIV,          {    0,    0,    0 } },
      { PimCmdEnum::AND,          {  128,   64,  128 } },
      { PimCmdEnum::OR,           {  128,   64,  128 } },
      { PimCmdEnum::XOR,          {  128,   64,  128 } },
      { PimCmdEnum::XNOR,         {  128,   64,  192 } },
      { PimCmdEnum::GT,           {  128,   64,  194 } },
      { PimCmdEnum::LT,           {  128,   64,  194 } },
      { PimCmdEnum::EQ,           {  128,   64,  195 } },
      { PimCmdEnum::MIN,          {  256,   64,  321 } },
      { PimCmdEnum::MAX,          {  256,   64,  321 } },
    }},
    { PIM_UINT8, {
      { PimCmdEnum::ABS,          {    8,    8,    0 } },
      //{ PimCmdEnum::POPCOUNT,     {    0,    0,    0 } },
      { PimCmdEnum::ADD,          {   16,    8,   25 } },
      { PimCmdEnum::SUB,          {   16,    8,   25 } },
      { PimCmdEnum::MUL,          {   72,   36,  136 } },
      { PimCmdEnum::DIV,          {  216,  140,  297 } },
      { PimCmdEnum::AND,          {   16,    8,   16 } },
      { PimCmdEnum::OR,           {   16,    8,   16 } },
      { PimCmdEnum::XOR,          {   16,    8,   16 } },
      { PimCmdEnum::XNOR,         {   16,    8,   24 } },
      { PimCmdEnum::GT,           {   16,    8,   27 } },
      { PimCmdEnum::LT,           {   16,    8,   27 } },
      { PimCmdEnum::EQ,           {   16,    8,   27 } },
      { PimCmdEnum::MIN,          {   32,    8,   42 } },
      { PimCmdEnum::MAX,          {   32,    8,   42 } },
    }},
    { PIM_UINT16, {
      { PimCmdEnum::ABS,          {   16,   16,    0 } },
      //{ PimCmdEnum::POPCOUNT,     {    0,    0,    0 } },
      { PimCmdEnum::ADD,          {   32,   16,   49 } },
      { PimCmdEnum::SUB,          {   32,   16,   49 } },
      { PimCmdEnum::MUL,          {  272,  136,  528 } },
      { PimCmdEnum::DIV,          {  816,  472, 1105 } },
      { PimCmdEnum::AND,          {   32,   16,   32 } },
      { PimCmdEnum::OR,           {   32,   16,   32 } },
      { PimCmdEnum::XOR,          {   32,   16,   32 } },
      { PimCmdEnum::XNOR,         {   32,   16,   48 } },
      { PimCmdEnum::GT,           {   32,   16,   51 } },
      { PimCmdEnum::LT,           {   32,   16,   51 } },
      { PimCmdEnum::EQ,           {   32,   16,   51 } },
      { PimCmdEnum::MIN,          {   64,   16,   82 } },
      { PimCmdEnum::MAX,          {   64,   16,   82 } },
    }},
    { PIM_UINT32, {
      { PimCmdEnum::ABS,          {   32,   32,    0 } },
      { PimCmdEnum::POPCOUNT,     {  114,  114,  218 } },
      { PimCmdEnum::ADD,          {   64,   32,   97 } },
      { PimCmdEnum::SUB,          {   64,   32,   97 } },
      { PimCmdEnum::MUL,          { 1056,  528, 2080 } },
      { PimCmdEnum::DIV,          { 3168, 1712, 4257 } },
      { PimCmdEnum::AND,          {   64,   32,   64 } },
      { PimCmdEnum::OR,           {   64,   32,   64 } },
      { PimCmdEnum::XOR,          {   64,   32,   64 } },
      { PimCmdEnum::XNOR,         {   64,   32,   96 } },
      { PimCmdEnum::GT,           {   64,   32,   99 } },
      { PimCmdEnum::LT,           {   64,   32,   99 } },
      { PimCmdEnum::EQ,           {   64,   32,   99 } },
      { PimCmdEnum::MIN,          {  128,   32,  162 } },
      { PimCmdEnum::MAX,          {  128,   32,  162 } },
    }},
    { PIM_UINT64, {
      { PimCmdEnum::ABS,          {   64,   64,    0 } },
      //{ PimCmdEnum::POPCOUNT,     {    0,    0,    0 } },
      { PimCmdEnum::ADD,          {  128,   64,  193 } },
      { PimCmdEnum::SUB,          {  128,   64,  193 } },
      //{ PimCmdEnum::MUL,          {    0,    0,    0 } },
      //{ PimCmdEnum::DIV,          {    0,    0,    0 } },
      { PimCmdEnum::AND,          {  128,   64,  128 } },
      { PimCmdEnum::OR,           {  128,   64,  128 } },
      { PimCmdEnum::XOR,          {  128,   64,  128 } },
      { PimCmdEnum::XNOR,         {  128,   64,  192 } },
      { PimCmdEnum::GT,           {  128,   64,  195 } },
      { PimCmdEnum::LT,           {  128,   64,  195 } },
      { PimCmdEnum::EQ,           {  128,   64,  195 } },
      { PimCmdEnum::MIN,          {  256,   64,  322 } },
      { PimCmdEnum::MAX,          {  256,   64,  322 } },
    }},
    { PIM_FP32, {
      { PimCmdEnum::ADD,          { 1331,  685, 1687 } },
      { PimCmdEnum::SUB,          { 1331,  685, 1687 } },
      { PimCmdEnum::MUL,          { 1852, 1000, 3054 } },
      { PimCmdEnum::DIV,          { 2744, 1458, 4187 } },
    }}
  }},
  { PIM_DEVICE_BITSIMD_V_AP, {
    { PIM_INT8, {
      { PimCmdEnum::ABS,          {    9,    8,   51 } },
      //{ PimCmdEnum::POPCOUNT,     {    0,    0,    0 } },
      { PimCmdEnum::ADD,          {   16,    8,   25 } },
      { PimCmdEnum::SUB,          {   16,    8,   25 } },
      { PimCmdEnum::MUL,          {   72,   36,  136 } },
      { PimCmdEnum::DIV,          {  196,  137,  493 } },
      { PimCmdEnum::AND,          {   16,    8,   16 } },
      { PimCmdEnum::OR,           {   16,    8,   17 } },
      { PimCmdEnum::XOR,          {   16,    8,   25 } },
      { PimCmdEnum::XNOR,         {   16,    8,   16 } },
      { PimCmdEnum::GT,           {   16,    8,   34 } },
      { PimCmdEnum::LT,           {   16,    8,   34 } },
      { PimCmdEnum::EQ,           {   16,    8,   27 } },
      { PimCmdEnum::MIN,          {   32,    8,   49 } },
      { PimCmdEnum::MAX,          {   32,    8,   49 } },
    }},
    { PIM_INT16, {
      { PimCmdEnum::ABS,          {   17,   16,   99 } },
      //{ PimCmdEnum::POPCOUNT,     {    0,    0,    0 } },
      { PimCmdEnum::ADD,          {   32,   16,   49 } },
      { PimCmdEnum::SUB,          {   32,   16,   49 } },
      { PimCmdEnum::MUL,          {  272,  136,  528 } },
      { PimCmdEnum::DIV,          {  772,  469, 1741 } },
      { PimCmdEnum::AND,          {   32,   16,   32 } },
      { PimCmdEnum::OR,           {   32,   16,   33 } },
      { PimCmdEnum::XOR,          {   32,   16,   49 } },
      { PimCmdEnum::XNOR,         {   32,   16,   32 } },
      { PimCmdEnum::GT,           {   32,   16,   66 } },
      { PimCmdEnum::LT,           {   32,   16,   66 } },
      { PimCmdEnum::EQ,           {   32,   16,   51 } },
      { PimCmdEnum::MIN,          {   64,   16,   97 } },
      { PimCmdEnum::MAX,          {   64,   16,   97 } },
    }},
    { PIM_INT32, {
      { PimCmdEnum::ABS,          {   33,   32,  195 } },
      { PimCmdEnum::POPCOUNT,     {  114,  114,  317 } },
      { PimCmdEnum::ADD,          {   64,   32,   97 } },
      { PimCmdEnum::SUB,          {   64,   32,   97 } },
      { PimCmdEnum::MUL,          { 1056,  528, 2080 } },
      { PimCmdEnum::DIV,          { 3076, 1709, 6541 } },
      { PimCmdEnum::AND,          {   64,   32,   64 } },
      { PimCmdEnum::OR,           {   64,   32,   65 } },
      { PimCmdEnum::XOR,          {   64,   32,   97 } },
      { PimCmdEnum::XNOR,         {   64,   32,   64 } },
      { PimCmdEnum::GT,           {   64,   32,  130 } },
      { PimCmdEnum::LT,           {   64,   32,  130 } },
      { PimCmdEnum::EQ,           {   64,   32,   99 } },
      { PimCmdEnum::MIN,          {  128,   32,  193 } },
      { PimCmdEnum::MAX,          {  128,   32,  193 } },
      // TODO: Scalers need to be updated with proper numbers. They are currently considering same #write & #logic as the non-scaler ones. But #read has been divided by 2
      //{ PimCmdEnum::ADD_SCALAR,   {   32,   33,  161 } },
      //{ PimCmdEnum::SUB_SCALAR,   {   32,   33,  161 } },
      //{ PimCmdEnum::MUL_SCALAR,   { 2146, 1799, 7039 } },
      //{ PimCmdEnum::DIV_SCALAR,   { 1864, 1744, 6800 } },
      //{ PimCmdEnum::AND_SCALAR,   {   32,   32,   64 } },
      //{ PimCmdEnum::OR_SCALAR,    {   32,   32,  128 } },
      //{ PimCmdEnum::XOR_SCALAR,   {   32,   32,  128 } },
      //{ PimCmdEnum::XNOR_SCALAR,  {   32,   32,   64 } },
      //{ PimCmdEnum::GT_SCALAR,    {   32,   32,   66 } },
      //{ PimCmdEnum::LT_SCALAR,    {   32,   32,   66 } },
      //{ PimCmdEnum::EQ_SCALAR,    {   32,   32,   66 } },
      //{ PimCmdEnum::MIN_SCALAR,   {   82,   67,  261 } },
      //{ PimCmdEnum::MAX_SCALAR,   {   82,   67,  261 } },
    }},
    { PIM_INT64, {
      { PimCmdEnum::ABS,          {   65,   64,  387 } },
      //{ PimCmdEnum::POPCOUNT,     {    0,    0,    0 } },
      { PimCmdEnum::ADD,          {  128,   64,  193 } },
      { PimCmdEnum::SUB,          {  128,   64,  193 } },
      //{ PimCmdEnum::MUL,          {    0,    0,    0 } },
      //{ PimCmdEnum::DIV,          {    0,    0,    0 } },
      { PimCmdEnum::AND,          {  128,   64,  128 } },
      { PimCmdEnum::OR,           {  128,   64,  129 } },
      { PimCmdEnum::XOR,          {  128,   64,  193 } },
      { PimCmdEnum::XNOR,         {  128,   64,  128 } },
      { PimCmdEnum::GT,           {  128,   64,  258 } },
      { PimCmdEnum::LT,           {  128,   64,  258 } },
      { PimCmdEnum::EQ,           {  128,   64,  195 } },
      { PimCmdEnum::MIN,          {  256,   64,  385 } },
      { PimCmdEnum::MAX,          {  256,   64,  385 } },
    }},
    { PIM_UINT8, {
      { PimCmdEnum::ABS,          {    8,    8,    0 } },
      { PimCmdEnum::POPCOUNT,     {    0,    0,    0 } },
      { PimCmdEnum::ADD,          {   16,    8,   25 } },
      { PimCmdEnum::SUB,          {   16,    8,   25 } },
      { PimCmdEnum::MUL,          {   72,   36,  136 } },
      { PimCmdEnum::DIV,          {  216,  140,  433 } },
      { PimCmdEnum::AND,          {   16,    8,   16 } },
      { PimCmdEnum::OR,           {   16,    8,   17 } },
      { PimCmdEnum::XOR,          {   16,    8,   25 } },
      { PimCmdEnum::XNOR,         {   16,    8,   16 } },
      { PimCmdEnum::GT,           {   16,    8,   36 } },
      { PimCmdEnum::LT,           {   16,    8,   36 } },
      { PimCmdEnum::EQ,           {   16,    8,   27 } },
      { PimCmdEnum::MIN,          {   32,    8,   51 } },
      { PimCmdEnum::MAX,          {   32,    8,   51 } },
    }},
    { PIM_UINT16, {
      { PimCmdEnum::ABS,          {   16,   16,    0 } },
      //{ PimCmdEnum::POPCOUNT,     {    0,    0,    0 } },
      { PimCmdEnum::ADD,          {   32,   16,   49 } },
      { PimCmdEnum::SUB,          {   32,   16,   49 } },
      { PimCmdEnum::MUL,          {  272,  136,  528 } },
      { PimCmdEnum::DIV,          {  816,  472, 1633 } },
      { PimCmdEnum::AND,          {   32,   16,   32 } },
      { PimCmdEnum::OR,           {   32,   16,   33 } },
      { PimCmdEnum::XOR,          {   32,   16,   49 } },
      { PimCmdEnum::XNOR,         {   32,   16,   32 } },
      { PimCmdEnum::GT,           {   32,   16,   68 } },
      { PimCmdEnum::LT,           {   32,   16,   68 } },
      { PimCmdEnum::EQ,           {   32,   16,   51 } },
      { PimCmdEnum::MIN,          {   64,   16,   99 } },
      { PimCmdEnum::MAX,          {   64,   16,   99 } },
    }},
    { PIM_UINT32, {
      { PimCmdEnum::ABS,          {   32,   32,    0 } },
      { PimCmdEnum::POPCOUNT,     {  114,  114,  317 } },
      { PimCmdEnum::ADD,          {   64,   32,   97 } },
      { PimCmdEnum::SUB,          {   64,   32,   97 } },
      { PimCmdEnum::MUL,          { 1056,  528, 2080 } },
      { PimCmdEnum::DIV,          { 3168, 1712, 6337 } },
      { PimCmdEnum::AND,          {   64,   32,   64 } },
      { PimCmdEnum::OR,           {   64,   32,   65 } },
      { PimCmdEnum::XOR,          {   64,   32,   97 } },
      { PimCmdEnum::XNOR,         {   64,   32,   64 } },
      { PimCmdEnum::GT,           {   64,   32,  132 } },
      { PimCmdEnum::LT,           {   64,   32,  132 } },
      { PimCmdEnum::EQ,           {   64,   32,   99 } },
      { PimCmdEnum::MIN,          {  128,   32,  195 } },
      { PimCmdEnum::MAX,          {  128,   32,  195 } },
    }},
    { PIM_UINT64, {
      { PimCmdEnum::ABS,          {   64,   64,    0 } },
      //{ PimCmdEnum::POPCOUNT,     {    0,    0,    0 } },
      { PimCmdEnum::ADD,          {  128,   64,  193 } },
      { PimCmdEnum::SUB,          {  128,   64,  193 } },
      //{ PimCmdEnum::MUL,          {    0,    0,    0 } },
      //{ PimCmdEnum::DIV,          {    0,    0,    0 } },
      { PimCmdEnum::AND,          {  128,   64,  128 } },
      { PimCmdEnum::OR,           {  128,   64,  129 } },
      { PimCmdEnum::XOR,          {  128,   64,  193 } },
      { PimCmdEnum::XNOR,         {  128,   64,  128 } },
      { PimCmdEnum::GT,           {  128,   64,  260 } },
      { PimCmdEnum::LT,           {  128,   64,  260 } },
      { PimCmdEnum::EQ,           {  128,   64,  195 } },
      { PimCmdEnum::MIN,          {  256,   64,  387 } },
      { PimCmdEnum::MAX,          {  256,   64,  387 } },
    }},
    { PIM_FP32, {
      { PimCmdEnum::ADD,          { 1597,  822, 2024 } },
      { PimCmdEnum::SUB,          { 1597,  822, 2024 } },
      { PimCmdEnum::MUL,          { 2222, 1200, 3664 } },
      { PimCmdEnum::DIV,          { 3292, 1749, 5024 } },
    }}
  }},
};

//! @brief  pimParamsPerf ctor
pimParamsPerf::pimParamsPerf(pimParamsDram* paramsDram)
  : m_paramsDram(paramsDram)
{
  m_tR = m_paramsDram->getNsRowRead() / 1000000.0;
  m_tW = m_paramsDram->getNsRowWrite() / 1000000.0;
  m_tL = m_paramsDram->getNsTCCD() / 1000000.0;
}

//! @brief  Set PIM device and simulation target
void
pimParamsPerf::setDevice(PimDeviceEnum deviceType)
{
  m_curDevice = deviceType;
  m_simTarget = deviceType;

  // determine simulation target for functional device
  if (deviceType == PIM_FUNCTIONAL) {
    PimDeviceEnum simTarget = PIM_DEVICE_NONE;
    // from 'make PIM_SIM_TARGET=...'
    #if defined(PIM_SIM_TARGET)
    simTarget = PIM_SIM_TARGET;
    #endif
    // default sim target
    if (simTarget == PIM_DEVICE_NONE || simTarget == PIM_FUNCTIONAL) {
      simTarget = PIM_DEVICE_BITSIMD_V;
    }
    m_simTarget = simTarget;
  }
}

//! @brief  If a PIM device uses vertical data layout
bool
pimParamsPerf::isVLayoutDevice() const
{
  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V: return true;
  case PIM_DEVICE_BITSIMD_V_AP: return true;
  case PIM_DEVICE_SIMDRAM: return true;
  case PIM_DEVICE_BITSIMD_H: return false;
  case PIM_DEVICE_FULCRUM: return false;
  case PIM_DEVICE_BANK_LEVEL: return false;
  case PIM_DEVICE_NONE:
  case PIM_FUNCTIONAL:
  default:
    assert(0);
  }
  return false;
}

//! @brief  If a PIM device uses horizontal data layout
bool
pimParamsPerf::isHLayoutDevice() const
{
  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V: return false;
  case PIM_DEVICE_BITSIMD_V_AP: return false;
  case PIM_DEVICE_SIMDRAM: return false;
  case PIM_DEVICE_BITSIMD_H: return true;
  case PIM_DEVICE_FULCRUM: return true;
  case PIM_DEVICE_BANK_LEVEL: return true;
  case PIM_DEVICE_NONE:
  case PIM_FUNCTIONAL:
  default:
    assert(0);
  }
  return false;
}

//! @brief  If a PIM device uses hybrid data layout
bool
pimParamsPerf::isHybridLayoutDevice() const
{
  return false;
}

//! @brief  Get ms runtime for bytes transferred between host and device
double
pimParamsPerf::getMsRuntimeForBytesTransfer(uint64_t numBytes) const
{
  int numRanks = static_cast<int>(pimSim::get()->getNumRanks());
  int numActiveRank = numRanks;
  double typicalRankBW = m_paramsDram->getTypicalRankBW(); // GB/s
  double totalMsRuntime = static_cast<double>(numBytes) / (typicalRankBW * numActiveRank * 1024 * 1024 * 1024 / 1000);
  return totalMsRuntime;
}

//! @brief  Get ms runtime for bit-serial PIM devices
//!         BitSIMD and SIMDRAM need different fields
double
pimParamsPerf::getMsRuntimeBitSerial(PimDeviceEnum deviceType, PimCmdEnum cmdType, PimDataType dataType, unsigned bitsPerElement, unsigned numPass) const
{
  const std::unordered_map<PimCmdEnum, PimCmdEnum> scalarCmdMap = {
    { PimCmdEnum::ADD_SCALAR, PimCmdEnum::ADD },
    { PimCmdEnum::SUB_SCALAR, PimCmdEnum::SUB },
    { PimCmdEnum::MUL_SCALAR, PimCmdEnum::MUL },
    { PimCmdEnum::DIV_SCALAR, PimCmdEnum::DIV },
    { PimCmdEnum::AND_SCALAR, PimCmdEnum::AND },
    { PimCmdEnum::OR_SCALAR, PimCmdEnum::OR },
    { PimCmdEnum::XOR_SCALAR, PimCmdEnum::XOR },
    { PimCmdEnum::XNOR_SCALAR, PimCmdEnum::XNOR },
    { PimCmdEnum::GT_SCALAR, PimCmdEnum::GT },
    { PimCmdEnum::LT_SCALAR, PimCmdEnum::LT },
    { PimCmdEnum::EQ_SCALAR, PimCmdEnum::EQ },
    { PimCmdEnum::MIN_SCALAR, PimCmdEnum::MIN },
    { PimCmdEnum::MAX_SCALAR, PimCmdEnum::MAX },
  };

  bool ok = false;
  double msRuntime = 0.0;

  // for scalar operand, add broadcast overhead for now
  if (scalarCmdMap.find(cmdType) != scalarCmdMap.end()) {
    msRuntime = (m_tL + m_tW) * bitsPerElement;
    msRuntime *= numPass;
    cmdType = scalarCmdMap.at(cmdType);
  }

  switch (deviceType) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
  case PIM_DEVICE_BITSIMD_H:
  {
    // BitSIMD-H reuse BitISMD-V perf for now
    if (deviceType == PIM_DEVICE_BITSIMD_H) {
      deviceType = PIM_DEVICE_BITSIMD_V;
    }
    // look up perf params from table
    auto it1 = s_bitsimdPerfTable.find(deviceType);
    if (it1 != s_bitsimdPerfTable.end()) {
      auto it2 = it1->second.find(dataType);
      if (it2 != it1->second.end()) {
        auto it3 = it2->second.find(cmdType);
        if (it3 != it2->second.end()) {
          const auto& [numR, numW, numL] = it3->second;
          msRuntime += m_tR * numR + m_tW * numW + m_tL * numL;
          ok = true;
        }
      }
    }
    // handle bit-shift specially
    if (cmdType == PimCmdEnum::SHIFT_BITS_L || cmdType == PimCmdEnum::SHIFT_BITS_R) {
      msRuntime += m_tR * (bitsPerElement - 1) + m_tW * bitsPerElement + m_tL;
      ok = true;
    }
    break;
  }
  case PIM_DEVICE_SIMDRAM:
  {
    break;
  }
  default:
    assert(0);
  }
  if (!ok) {
    std::printf("PIM-Warning: Unimplemented bit-serial runtime estimation for device=%s cmd=%s dataType=%s\n",
            pimUtils::pimDeviceEnumToStr(deviceType).c_str(),
            pimCmd::getName(cmdType, "").c_str(),
            pimUtils::pimDataTypeEnumToStr(dataType).c_str());
    msRuntime = 1000000;
  }
  msRuntime *= numPass;
  return msRuntime;
}

//! @brief  Get ms runtime for func1
double
pimParamsPerf::getMsRuntimeForFunc1(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  PimDataType dataType = obj.getDataType();
  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
  case PIM_DEVICE_BITSIMD_H:
  case PIM_DEVICE_SIMDRAM:
    msRuntime += getMsRuntimeBitSerial(m_simTarget, cmdType, dataType, bitsPerElement, numPass);
    break;
  case PIM_DEVICE_FULCRUM:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double aluLatency = 0.000005; // 5ns
    unsigned aluBits = 32; // 32-bit ALU
    double numberOfALUOperationPerCycle = ((double)bitsPerElement/aluBits);
    msRuntime = m_tR + m_tW + maxElementsPerRegion * aluLatency * numberOfALUOperationPerCycle * numPass;
    switch (cmdType)
    {
    case PimCmdEnum::ADD_SCALAR:
    case PimCmdEnum::SUB_SCALAR:
    case PimCmdEnum::MUL_SCALAR:
    case PimCmdEnum::DIV_SCALAR:
    case PimCmdEnum::AND_SCALAR:
    case PimCmdEnum::OR_SCALAR:
    case PimCmdEnum::XOR_SCALAR:
    case PimCmdEnum::XNOR_SCALAR:
    case PimCmdEnum::GT_SCALAR:
    case PimCmdEnum::LT_SCALAR:
    case PimCmdEnum::EQ_SCALAR:
    case PimCmdEnum::MIN_SCALAR:
    case PimCmdEnum::MAX_SCALAR: msRuntime += aluLatency * maxElementsPerRegion; break; // the broadcast value is being stored in the walker, hence no row write is needed.
    case PimCmdEnum::POPCOUNT: msRuntime *= 12; break; // 4 shifts, 4 ands, 3 add/sub, 1 mul
    case PimCmdEnum::ABS:
    case PimCmdEnum::SHIFT_BITS_L:
    case PimCmdEnum::SHIFT_BITS_R: break;
    default:
       std::printf("PIM-Warning: Unsupported PIM command.\n");
       break;
    }
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double aluLatency = 0.000005; // 5ns
    unsigned numALU = 2;
    unsigned bitsPerElement = obj.getBitsPerElement();
    unsigned aluBits = 32; // 32-bit ALU
    double numberOfALUOperationPerCycle = ((double)bitsPerElement/aluBits);
    msRuntime =  m_tR + m_tW + maxElementsPerRegion * aluLatency * numberOfALUOperationPerCycle * numPass / numALU;
    switch (cmdType)
    {
    case PimCmdEnum::POPCOUNT:
    case PimCmdEnum::ABS:
    case PimCmdEnum::SHIFT_BITS_L:
    case PimCmdEnum::SHIFT_BITS_R: break;
    case PimCmdEnum::ADD_SCALAR:
    case PimCmdEnum::SUB_SCALAR:
    case PimCmdEnum::MUL_SCALAR:
    case PimCmdEnum::DIV_SCALAR:
    case PimCmdEnum::AND_SCALAR:
    case PimCmdEnum::OR_SCALAR:
    case PimCmdEnum::XOR_SCALAR:
    case PimCmdEnum::XNOR_SCALAR:
    case PimCmdEnum::GT_SCALAR:
    case PimCmdEnum::LT_SCALAR:
    case PimCmdEnum::EQ_SCALAR:
    case PimCmdEnum::MIN_SCALAR:
    case PimCmdEnum::MAX_SCALAR: msRuntime += aluLatency * maxElementsPerRegion; break; // the broadcast value is being stored in the V0 register, hence no row write is needed.
    default:
       std::printf("PIM-Warning: Unsupported PIM command.\n");
       break;
    }
    break;
  }
  default:
    msRuntime = 1000000;
  }
  return msRuntime;
}

//! @brief  Get ms runtime for func2
double
pimParamsPerf::getMsRuntimeForFunc2(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  PimDataType dataType = obj.getDataType();

  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
  case PIM_DEVICE_BITSIMD_H:
  case PIM_DEVICE_SIMDRAM:
    msRuntime = getMsRuntimeBitSerial(m_simTarget, cmdType, dataType, bitsPerElement, numPass);
    break;
  case PIM_DEVICE_FULCRUM:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double aluLatency = 0.000005; // 5ns
    unsigned aluBits = 32; // 32-bit ALU
    double numberOfALUOperationPerCycle = (bitsPerElement/aluBits);
    msRuntime = 2 * m_tR + m_tW + maxElementsPerRegion * numberOfALUOperationPerCycle * aluLatency;
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double aluLatency = 0.000005; // 5ns
    unsigned numALU = 2;
    unsigned bitsPerElement = obj.getBitsPerElement();
    unsigned aluBits = 32; // 32-bit ALU
    double numberOfALUOperationPerCycle = (bitsPerElement/aluBits);
    msRuntime = 2 * m_tR + m_tW + maxElementsPerRegion * aluLatency * numberOfALUOperationPerCycle / numALU;
    msRuntime *= numPass;
    break;
  }
  default:
    msRuntime = 1e10;
  }
  return msRuntime;
}

//! @brief  Get ms runtime for reduction sum
double
pimParamsPerf::getMsRuntimeForRedSum(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  PimDataType dataType = obj.getDataType();
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numRegions = obj.getRegions().size();
  uint64_t numElements = obj.getNumElements();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();

  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
    if (dataType == PIM_INT32 || dataType == PIM_UINT32) {
      // Assume pop count reduction circut in tR runtime
      msRuntime = ((m_tR + m_tR) * bitsPerElement);
      msRuntime *= numPass;
      // reduction for all regions
      msRuntime += static_cast<double>(numRegions) / 3200000;
    } else if (dataType == PIM_INT8 || dataType == PIM_INT16 || dataType == PIM_INT64 || dataType == PIM_UINT8 || dataType == PIM_UINT16 || dataType == PIM_UINT64) {
      // todo
      std::printf("PIM-Warning: BitSIMD int & uint 8/16/64 performance stats not implemented yet.\n");
    } else {
      assert(0);
    }
    break;
  case PIM_DEVICE_SIMDRAM:
    // todo
    std::printf("PIM-Warning: SIMDRAM performance stats not implemented yet.\n");
    break;
  case PIM_DEVICE_BITSIMD_H:
    // Sequentially process all elements per CPU cycle
    msRuntime = static_cast<double>(numElements) / 3200000; // typical 3.2 GHz CPU
    // consider PCL
    break;
  case PIM_DEVICE_FULCRUM:
  case PIM_DEVICE_BANK_LEVEL:
  {
    // read a row to walker, then reduce in serial
    double aluLatency = 0.000005; // 5ns
    msRuntime = (m_tR + maxElementsPerRegion * aluLatency);
    msRuntime *= numPass;
    // reduction for all regions
    msRuntime += static_cast<double>(numRegions) / 3200000;
    break;
  }
  default:
    msRuntime = 1e10;
  }

  return msRuntime;
}

//! @brief  Get ms runtime for broadcast
double
pimParamsPerf::getMsRuntimeForBroadcast(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();

  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
  {
    // For one pass: For every bit: Set SA to bit value; Write SA to row;
    msRuntime = (m_tL + m_tW) * bitsPerElement;
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_SIMDRAM:
  {
    // todo
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_BITSIMD_H:
  {
    // For one pass: For every element: 1 tCCD per byte
    uint64_t maxBytesPerRegion = (uint64_t)maxElementsPerRegion * (bitsPerElement / 8);
    msRuntime = m_tW + m_tL * maxBytesPerRegion; // for one pass
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_FULCRUM:
  case PIM_DEVICE_BANK_LEVEL:
  {
    // assume taking 1 ALU latency to write an element
    double aluLatency = 0.000005; // 5ns
    msRuntime = m_tW + aluLatency * maxElementsPerRegion;
    msRuntime *= numPass;
    break;
  }
  default:
    msRuntime = 1e10;
  }

  return msRuntime;
}

//! @brief  Get ms runtime for rotate
double
//pimParamsPerf::getMsRuntimeForRotate(PimCmdEnum cmdType, unsigned bitsPerElement, unsigned numRegions) const
pimParamsPerf::getMsRuntimeForRotate(PimCmdEnum cmdType, const pimObjInfo& obj) const
{
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numRegions = obj.getRegions().size();

  switch (m_simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
    // rotate within subarray:
    // For every bit: Read row to SA; move SA to R1; Shift R1; Move R1 to SA; Write SA to row
    msRuntime = (m_tR + 3 * m_tL + m_tW) * bitsPerElement; // for one pass
    msRuntime *= numPass;
    // boundary handling
    msRuntime += 2 * getMsRuntimeForBytesTransfer(numRegions * bitsPerElement / 8);
    break;
  case PIM_DEVICE_SIMDRAM:
    // todo
    break;
  case PIM_DEVICE_BITSIMD_H:
  case PIM_DEVICE_FULCRUM:
  case PIM_DEVICE_BANK_LEVEL:
    // rotate within subarray:
    // For every bit: Read row to SA; move SA to R1; Shift R1 by N steps; Move R1 to SA; Write SA to row
    msRuntime = (m_tR + (bitsPerElement + 2) * m_tL + m_tW); // for one pass
    msRuntime *= numPass;
    // boundary handling
    msRuntime += 2 * getMsRuntimeForBytesTransfer(numRegions * bitsPerElement / 8);
    break;
  default:
    msRuntime = 1e10;
  }

  return msRuntime;
}

