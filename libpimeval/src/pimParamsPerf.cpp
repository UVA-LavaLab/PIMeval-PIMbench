// File: pimParamsPerf.cc
// PIMeval Simulator - Performance parameters
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

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
      { PimCmdEnum::ADD_SCALAR,   {    8,    8,   33 } },
      { PimCmdEnum::SUB_SCALAR,   {    8,    8,   33 } },
      { PimCmdEnum::MUL_SCALAR,   {   36,   36,  172 } },
      { PimCmdEnum::DIV_SCALAR,   {  146,  145,  394 } },
      { PimCmdEnum::AND_SCALAR,   {    8,    8,   24 } },
      { PimCmdEnum::OR_SCALAR,    {    8,    8,   24 } },
      { PimCmdEnum::XOR_SCALAR,   {    8,    8,   24 } },
      { PimCmdEnum::XNOR_SCALAR,  {    8,    8,   32 } },
      { PimCmdEnum::GT_SCALAR,    {    8,    8,   34 } },
      { PimCmdEnum::LT_SCALAR,    {    8,    8,   34 } },
      { PimCmdEnum::EQ_SCALAR,    {    8,    8,   35 } },
      { PimCmdEnum::MIN_SCALAR,   {   16,    8,   57 } },
      { PimCmdEnum::MAX_SCALAR,   {   16,    8,   57 } },
      { PimCmdEnum::SCALED_ADD,{   44,   44,  197 } }, // Derived from adding ADD + MUL_SCALAR 
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
      { PimCmdEnum::ADD_SCALAR,   {   16,   16,   65 } },
      { PimCmdEnum::SUB_SCALAR,   {   16,   16,   65 } },
      { PimCmdEnum::MUL_SCALAR,   {  136,  136,  664 } },
      { PimCmdEnum::DIV_SCALAR,   {  546,  485, 1418 } },
      { PimCmdEnum::AND_SCALAR,   {   16,   16,   48 } },
      { PimCmdEnum::OR_SCALAR,    {   16,   16,   48 } },
      { PimCmdEnum::XOR_SCALAR,   {   16,   16,   48 } },
      { PimCmdEnum::XNOR_SCALAR,  {   16,   16,   64 } },
      { PimCmdEnum::GT_SCALAR,    {   16,   16,   66 } },
      { PimCmdEnum::LT_SCALAR,    {   16,   16,   66 } },
      { PimCmdEnum::EQ_SCALAR,    {   16,   16,   67 } },
      { PimCmdEnum::MIN_SCALAR,   {   32,   16,  113 } },
      { PimCmdEnum::MAX_SCALAR,   {   32,   16,  113 } },
      { PimCmdEnum::SCALED_ADD,{  168,  152,  713 } }, // Derived from adding ADD + MUL_SCALAR 
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
      { PimCmdEnum::ADD_SCALAR,   {   32,   32,  129 } },
      { PimCmdEnum::SUB_SCALAR,   {   32,   32,  129 } },
      { PimCmdEnum::MUL_SCALAR,   {  528,  528, 2608 } },
      { PimCmdEnum::DIV_SCALAR,   { 2114, 1741, 5386 } },
      { PimCmdEnum::AND_SCALAR,   {   32,   32,   96 } },
      { PimCmdEnum::OR_SCALAR,    {   32,   32,   96 } },
      { PimCmdEnum::XOR_SCALAR,   {   32,   32,   96 } },
      { PimCmdEnum::XNOR_SCALAR,  {   32,   32,  128 } },
      { PimCmdEnum::GT_SCALAR,    {   32,   32,  130 } },
      { PimCmdEnum::LT_SCALAR,    {   32,   32,  130 } },
      { PimCmdEnum::EQ_SCALAR,    {   32,   32,  131 } },
      { PimCmdEnum::MIN_SCALAR,   {   64,   32,  225 } },
      { PimCmdEnum::MAX_SCALAR,   {   64,   32,  225 } },
      { PimCmdEnum::SCALED_ADD,{  592,  560, 2705 } }, // Derived from adding ADD + MUL_SCALAR 
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
      { PimCmdEnum::ADD_SCALAR,   {   64,   64,  257 } },
      { PimCmdEnum::SUB_SCALAR,   {   64,   64,  257 } },
      //{ PimCmdEnum::MUL_SCALAR,   {    0,    0,    0 } },
      //{ PimCmdEnum::DIV_SCALAR,   {    0,    0,    0 } },
      { PimCmdEnum::AND_SCALAR,   {   64,   64,  192 } },
      { PimCmdEnum::OR_SCALAR,    {   64,   64,  192 } },
      { PimCmdEnum::XOR_SCALAR,   {   64,   64,  192 } },
      { PimCmdEnum::XNOR_SCALAR,  {   64,   64,  256 } },
      { PimCmdEnum::GT_SCALAR,    {   64,   64,  258 } },
      { PimCmdEnum::LT_SCALAR,    {   64,   64,  258 } },
      { PimCmdEnum::EQ_SCALAR,    {   64,   64,  259 } },
      { PimCmdEnum::MIN_SCALAR,   {  128,   64,  449 } },
      { PimCmdEnum::MAX_SCALAR,   {  128,   64,  449 } },
      //{ PimCmdEnum::MUL_AGGREGATE,{  592,  560, 2705 } }, // Derived from adding ADD + MUL_SCALAR 
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
      { PimCmdEnum::ADD_SCALAR,   {    8,    8,   33 } },
      { PimCmdEnum::SUB_SCALAR,   {    8,    8,   33 } },
      { PimCmdEnum::MUL_SCALAR,   {   36,   36,  172 } },
      { PimCmdEnum::DIV_SCALAR,   {  152,  140,  361 } },
      { PimCmdEnum::AND_SCALAR,   {    8,    8,   24 } },
      { PimCmdEnum::OR_SCALAR,    {    8,    8,   24 } },
      { PimCmdEnum::XOR_SCALAR,   {    8,    8,   24 } },
      { PimCmdEnum::XNOR_SCALAR,  {    8,    8,   32 } },
      { PimCmdEnum::GT_SCALAR,    {    8,    8,   35 } },
      { PimCmdEnum::LT_SCALAR,    {    8,    8,   35 } },
      { PimCmdEnum::EQ_SCALAR,    {    8,    8,   35 } },
      { PimCmdEnum::MIN_SCALAR,   {   16,    8,   58 } },
      { PimCmdEnum::MAX_SCALAR,   {   16,    8,   58 } },
      { PimCmdEnum::SCALED_ADD,{   44,   44,  197 } }, // Derived from adding ADD + MUL_SCALAR 
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
      { PimCmdEnum::ADD_SCALAR,   {   16,   16,   65 } },
      { PimCmdEnum::SUB_SCALAR,   {   16,   16,   65 } },
      { PimCmdEnum::MUL_SCALAR,   {  136,  136,  664 } },
      { PimCmdEnum::DIV_SCALAR,   {  560,  472, 1361 } },
      { PimCmdEnum::AND_SCALAR,   {   16,   16,   48 } },
      { PimCmdEnum::OR_SCALAR,    {   16,   16,   48 } },
      { PimCmdEnum::XOR_SCALAR,   {   16,   16,   48 } },
      { PimCmdEnum::XNOR_SCALAR,  {   16,   16,   64 } },
      { PimCmdEnum::GT_SCALAR,    {   16,   16,   67 } },
      { PimCmdEnum::LT_SCALAR,    {   16,   16,   67 } },
      { PimCmdEnum::EQ_SCALAR,    {   16,   16,   67 } },
      { PimCmdEnum::MIN_SCALAR,   {   32,   16,  114 } },
      { PimCmdEnum::MAX_SCALAR,   {   32,   16,  114 } },
      { PimCmdEnum::SCALED_ADD,{  168,  152,  713 } }, // Derived from adding ADD + MUL_SCALAR 
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
      { PimCmdEnum::ADD_SCALAR,   {   32,   32,  129 } },
      { PimCmdEnum::SUB_SCALAR,   {   32,   32,  129 } },
      { PimCmdEnum::MUL_SCALAR,   {  528,  528, 2608 } },
      { PimCmdEnum::DIV_SCALAR,   { 2144, 1712, 5281 } },
      { PimCmdEnum::AND_SCALAR,   {   32,   32,   96 } },
      { PimCmdEnum::OR_SCALAR,    {   32,   32,   96 } },
      { PimCmdEnum::XOR_SCALAR,   {   32,   32,   96 } },
      { PimCmdEnum::XNOR_SCALAR,  {   32,   32,  128 } },
      { PimCmdEnum::GT_SCALAR,    {   32,   32,  131 } },
      { PimCmdEnum::LT_SCALAR,    {   32,   32,  131 } },
      { PimCmdEnum::EQ_SCALAR,    {   32,   32,  131 } },
      { PimCmdEnum::MIN_SCALAR,   {   64,   32,  226 } },
      { PimCmdEnum::MAX_SCALAR,   {   64,   32,  226 } },
      { PimCmdEnum::SCALED_ADD,{  592,  560, 2705 } }, // Derived from adding ADD + MUL_SCALAR 
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
      { PimCmdEnum::ADD_SCALAR,   {   64,   64,  257 } },
      { PimCmdEnum::SUB_SCALAR,   {   64,   64,  257 } },
      //{ PimCmdEnum::MUL_SCALAR,   {    0,    0,    0 } },
      //{ PimCmdEnum::DIV_SCALAR,   {    0,    0,    0 } },
      { PimCmdEnum::AND_SCALAR,   {   64,   64,  192 } },
      { PimCmdEnum::OR_SCALAR,    {   64,   64,  192 } },
      { PimCmdEnum::XOR_SCALAR,   {   64,   64,  192 } },
      { PimCmdEnum::XNOR_SCALAR,  {   64,   64,  256 } },
      { PimCmdEnum::GT_SCALAR,    {   64,   64,  259 } },
      { PimCmdEnum::LT_SCALAR,    {   64,   64,  259 } },
      { PimCmdEnum::EQ_SCALAR,    {   64,   64,  259 } },
      { PimCmdEnum::MIN_SCALAR,   {  128,   64,  450 } },
      { PimCmdEnum::MAX_SCALAR,   {  128,   64,  450 } },
      // { PimCmdEnum::SCALED_ADD,{  592,  560, 2705 } }, // Derived from adding ADD + MUL_SCALAR 
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
      { PimCmdEnum::ADD_SCALAR,   {    8,    8,   33 } },
      { PimCmdEnum::SUB_SCALAR,   {    8,    8,   33 } },
      { PimCmdEnum::MUL_SCALAR,   {   36,   36,  172 } },
      { PimCmdEnum::DIV_SCALAR,   {  146,  145,  551 } },
      { PimCmdEnum::AND_SCALAR,   {    8,    8,   24 } },
      { PimCmdEnum::OR_SCALAR,    {    8,    8,   25 } },
      { PimCmdEnum::XOR_SCALAR,   {    8,    8,   33 } },
      { PimCmdEnum::XNOR_SCALAR,  {    8,    8,   24 } },
      { PimCmdEnum::GT_SCALAR,    {    8,    8,   42 } },
      { PimCmdEnum::LT_SCALAR,    {    8,    8,   42 } },
      { PimCmdEnum::EQ_SCALAR,    {    8,    8,   35 } },
      { PimCmdEnum::MIN_SCALAR,   {   16,    8,   65 } },
      { PimCmdEnum::MAX_SCALAR,   {   16,    8,   65 } },
      { PimCmdEnum::SCALED_ADD,{   52,   44,  197 } }, // Derived from adding ADD + MUL_SCALAR 
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
      { PimCmdEnum::ADD_SCALAR,   {   16,   16,   65 } },
      { PimCmdEnum::SUB_SCALAR,   {   16,   16,   65 } },
      { PimCmdEnum::MUL_SCALAR,   {  136,  136,  664 } },
      { PimCmdEnum::DIV_SCALAR,   {  546,  485, 1983 } },
      { PimCmdEnum::AND_SCALAR,   {   16,   16,   48 } },
      { PimCmdEnum::OR_SCALAR,    {   16,   16,   49 } },
      { PimCmdEnum::XOR_SCALAR,   {   16,   16,   65 } },
      { PimCmdEnum::XNOR_SCALAR,  {   16,   16,   48 } },
      { PimCmdEnum::GT_SCALAR,    {   16,   16,   82 } },
      { PimCmdEnum::LT_SCALAR,    {   16,   16,   82 } },
      { PimCmdEnum::EQ_SCALAR,    {   16,   16,   67 } },
      { PimCmdEnum::MIN_SCALAR,   {   32,   16,  129 } },
      { PimCmdEnum::MAX_SCALAR,   {   32,   16,  129 } },
      { PimCmdEnum::SCALED_ADD,{  168,  152,  713 } }, // Derived from adding ADD + MUL_SCALAR 
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
      { PimCmdEnum::ADD_SCALAR,   {   32,   32,  129 } },
      { PimCmdEnum::SUB_SCALAR,   {   32,   32,  129 } },
      { PimCmdEnum::MUL_SCALAR,   {  528,  528, 2608 } },
      { PimCmdEnum::DIV_SCALAR,   { 2114, 1741, 7535 } },
      { PimCmdEnum::AND_SCALAR,   {   32,   32,   96 } },
      { PimCmdEnum::OR_SCALAR,    {   32,   32,   97 } },
      { PimCmdEnum::XOR_SCALAR,   {   32,   32,  129 } },
      { PimCmdEnum::XNOR_SCALAR,  {   32,   32,   96 } },
      { PimCmdEnum::GT_SCALAR,    {   32,   32,  162 } },
      { PimCmdEnum::LT_SCALAR,    {   32,   32,  162 } },
      { PimCmdEnum::EQ_SCALAR,    {   32,   32,  131 } },
      { PimCmdEnum::MIN_SCALAR,   {   64,   32,  257 } },
      { PimCmdEnum::MAX_SCALAR,   {   64,   32,  257 } },
      { PimCmdEnum::SCALED_ADD,{  592,  560, 2705 } }, // Derived from adding ADD + MUL_SCALAR 
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
      { PimCmdEnum::ADD_SCALAR,   {   64,   64,  257 } },
      { PimCmdEnum::SUB_SCALAR,   {   64,   64,  257 } },
      //{ PimCmdEnum::MUL_SCALAR,   {    0,    0,    0 } },
      //{ PimCmdEnum::DIV_SCALAR,   {    0,    0,    0 } },
      { PimCmdEnum::AND_SCALAR,   {   64,   64,  192 } },
      { PimCmdEnum::OR_SCALAR,    {   64,   64,  193 } },
      { PimCmdEnum::XOR_SCALAR,   {   64,   64,  257 } },
      { PimCmdEnum::XNOR_SCALAR,  {   64,   64,  192 } },
      { PimCmdEnum::GT_SCALAR,    {   64,   64,  322 } },
      { PimCmdEnum::LT_SCALAR,    {   64,   64,  322 } },
      { PimCmdEnum::EQ_SCALAR,    {   64,   64,  259 } },
      { PimCmdEnum::MIN_SCALAR,   {  128,   64,  513 } },
      { PimCmdEnum::MAX_SCALAR,   {  128,   64,  513 } },
      // { PimCmdEnum::SCALED_ADD,{   52,   44,  197 } }, // Derived from adding ADD + MUL_SCALAR 
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
      { PimCmdEnum::ADD_SCALAR,   {    8,    8,   33 } },
      { PimCmdEnum::SUB_SCALAR,   {    8,    8,   33 } },
      { PimCmdEnum::MUL_SCALAR,   {   36,   36,  172 } },
      { PimCmdEnum::DIV_SCALAR,   {  152,  140,  497 } },
      { PimCmdEnum::AND_SCALAR,   {    8,    8,   24 } },
      { PimCmdEnum::OR_SCALAR,    {    8,    8,   25 } },
      { PimCmdEnum::XOR_SCALAR,   {    8,    8,   33 } },
      { PimCmdEnum::XNOR_SCALAR,  {    8,    8,   24 } },
      { PimCmdEnum::GT_SCALAR,    {    8,    8,   44 } },
      { PimCmdEnum::LT_SCALAR,    {    8,    8,   44 } },
      { PimCmdEnum::EQ_SCALAR,    {    8,    8,   35 } },
      { PimCmdEnum::MIN_SCALAR,   {   16,    8,   67 } },
      { PimCmdEnum::MAX_SCALAR,   {   16,    8,   67 } },
      { PimCmdEnum::SCALED_ADD,{   52,   44,  197 } }, // Derived from adding ADD + MUL_SCALAR 
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
      { PimCmdEnum::ADD_SCALAR,   {   16,   16,   65 } },
      { PimCmdEnum::SUB_SCALAR,   {   16,   16,   65 } },
      { PimCmdEnum::MUL_SCALAR,   {  136,  136,  664 } },
      { PimCmdEnum::DIV_SCALAR,   {  560,  472, 1889 } },
      { PimCmdEnum::AND_SCALAR,   {   16,   16,   48 } },
      { PimCmdEnum::OR_SCALAR,    {   16,   16,   49 } },
      { PimCmdEnum::XOR_SCALAR,   {   16,   16,   65 } },
      { PimCmdEnum::XNOR_SCALAR,  {   16,   16,   48 } },
      { PimCmdEnum::GT_SCALAR,    {   16,   16,   84 } },
      { PimCmdEnum::LT_SCALAR,    {   16,   16,   84 } },
      { PimCmdEnum::EQ_SCALAR,    {   16,   16,   67 } },
      { PimCmdEnum::MIN_SCALAR,   {   32,   16,  131 } },
      { PimCmdEnum::MAX_SCALAR,   {   32,   16,  131 } },
      { PimCmdEnum::SCALED_ADD,{  168,  152,  713 } }, // Derived from adding ADD + MUL_SCALAR 
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
      { PimCmdEnum::ADD_SCALAR,   {   32,   32,  129 } },
      { PimCmdEnum::SUB_SCALAR,   {   32,   32,  129 } },
      { PimCmdEnum::MUL_SCALAR,   {  528,  528, 2608 } },
      { PimCmdEnum::DIV_SCALAR,   { 2144, 1712, 7361 } },
      { PimCmdEnum::AND_SCALAR,   {   32,   32,   96 } },
      { PimCmdEnum::OR_SCALAR,    {   32,   32,   97 } },
      { PimCmdEnum::XOR_SCALAR,   {   32,   32,  129 } },
      { PimCmdEnum::XNOR_SCALAR,  {   32,   32,   96 } },
      { PimCmdEnum::GT_SCALAR,    {   32,   32,  164 } },
      { PimCmdEnum::LT_SCALAR,    {   32,   32,  164 } },
      { PimCmdEnum::EQ_SCALAR,    {   32,   32,  131 } },
      { PimCmdEnum::MIN_SCALAR,   {   64,   32,  259 } },
      { PimCmdEnum::MAX_SCALAR,   {   64,   32,  259 } },
      { PimCmdEnum::SCALED_ADD,{  592,  560, 2705 } }, // Derived from adding ADD + MUL_SCALAR 
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
      { PimCmdEnum::ADD_SCALAR,   {   64,   64,  257 } },
      { PimCmdEnum::SUB_SCALAR,   {   64,   64,  257 } },
      //{ PimCmdEnum::MUL_SCALAR,   {    0,    0,    0 } },
      //{ PimCmdEnum::DIV_SCALAR,   {    0,    0,    0 } },
      { PimCmdEnum::AND_SCALAR,   {   64,   64,  192 } },
      { PimCmdEnum::OR_SCALAR,    {   64,   64,  193 } },
      { PimCmdEnum::XOR_SCALAR,   {   64,   64,  257 } },
      { PimCmdEnum::XNOR_SCALAR,  {   64,   64,  192 } },
      { PimCmdEnum::GT_SCALAR,    {   64,   64,  324 } },
      { PimCmdEnum::LT_SCALAR,    {   64,   64,  324 } },
      { PimCmdEnum::EQ_SCALAR,    {   64,   64,  259 } },
      { PimCmdEnum::MIN_SCALAR,   {  128,   64,  515 } },
      { PimCmdEnum::MAX_SCALAR,   {  128,   64,  515 } },
      // { PimCmdEnum::MUL_AGGREGATE,{  592,  560, 2705 } }, // Derived from adding ADD + MUL_SCALAR 
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
  bool ok = false;
  double msRuntime = 0.0;

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
  PimDeviceEnum simTarget = pimSim::get()->getSimTarget();
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  PimDataType dataType = obj.getDataType();
  switch (simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
  case PIM_DEVICE_BITSIMD_H:
  case PIM_DEVICE_SIMDRAM:
    msRuntime += getMsRuntimeBitSerial(simTarget, cmdType, dataType, bitsPerElement, numPass);
    break;
  case PIM_DEVICE_FULCRUM:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double numberOfALUOperationPerElement = ((double)bitsPerElement / m_flucrumAluBitWidth); 
    switch (cmdType)
    {
    case PimCmdEnum::POPCOUNT: numberOfALUOperationPerElement *= 12; break; // 4 shifts, 4 ands, 3 add/sub, 1 mul
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
    case PimCmdEnum::MAX_SCALAR:
    case PimCmdEnum::ABS:
    case PimCmdEnum::SHIFT_BITS_L:
    case PimCmdEnum::SHIFT_BITS_R: break;
    default:
       std::printf("PIM-Warning: Unsupported PIM command.\n");
       break;
    }

    // Fulcrum utilizes three walkers: two for input operands and one for the output operand.
    // For instructions that operate on a single operand, the next operand is fetched by the walker.
    // Consequently, only one row read operation is required in this case.
    // Additionally, using the walker-renaming technique (refer to the Fulcrum paper for details),
    // the write operation is also pipelined. Thus, only one row write operation is needed.
    msRuntime = m_tR + m_tW + (maxElementsPerRegion * m_fulcrumAluLatency * numberOfALUOperationPerElement * numPass);
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
    // Refer to fulcrum documentation
    msRuntime = m_tR + m_tW + (maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement * numPass);
    switch (cmdType)
    {
    case PimCmdEnum::POPCOUNT:
    case PimCmdEnum::ABS:
    case PimCmdEnum::SHIFT_BITS_L:
    case PimCmdEnum::SHIFT_BITS_R:
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
    case PimCmdEnum::MAX_SCALAR: break;
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
  PimDeviceEnum simTarget = pimSim::get()->getSimTarget();
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  PimDataType dataType = obj.getDataType();

  switch (simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
  case PIM_DEVICE_BITSIMD_H:
  case PIM_DEVICE_SIMDRAM:
    msRuntime = getMsRuntimeBitSerial(simTarget, cmdType, dataType, bitsPerElement, numPass);
    break;
  case PIM_DEVICE_FULCRUM:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double numberOfALUOperationPerElement = ((double)bitsPerElement / m_flucrumAluBitWidth);
    msRuntime = 2 * m_tR + m_tW + maxElementsPerRegion * numberOfALUOperationPerElement * m_fulcrumAluLatency;
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
    double numberOfALUOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
    msRuntime = 2 * m_tR + m_tW + maxElementsPerRegion * m_blimpCoreLatency * numberOfALUOperationPerElement;
    switch (cmdType)
    {
    case PimCmdEnum::SCALED_ADD:
    {
      msRuntime += maxElementsPerRegion * numberOfALUOperationPerElement * m_blimpCoreLatency;
      break;
    }
    case PimCmdEnum::ADD:
    case PimCmdEnum::SUB:
    case PimCmdEnum::MUL:
    case PimCmdEnum::DIV:
    case PimCmdEnum::AND:
    case PimCmdEnum::OR:
    case PimCmdEnum::XOR:
    case PimCmdEnum::XNOR:
    case PimCmdEnum::GT:
    case PimCmdEnum::LT:
    case PimCmdEnum::EQ:
    case PimCmdEnum::MIN:
    case PimCmdEnum::MAX:  break;
    default:
       std::printf("PIM-Warning: Unsupported PIM command.\n");
       break;
    }
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
pimParamsPerf::getMsRuntimeForRedSum(PimCmdEnum cmdType, const pimObjInfo& obj, unsigned numPass) const
{
  PimDeviceEnum simTarget = pimSim::get()->getSimTarget();
  double msRuntime = 0.0;
  PimDataType dataType = obj.getDataType();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numRegions = obj.getRegions().size();
  uint64_t numElements = obj.getNumElements();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();
  unsigned numCore = obj.getNumCoresUsed();

  switch (simTarget) {
  case PIM_DEVICE_BITSIMD_V:
  case PIM_DEVICE_BITSIMD_V_AP:
    if (dataType == PIM_INT8 || dataType == PIM_INT16 || dataType == PIM_INT64 || dataType == PIM_INT32 || dataType == PIM_UINT8 || dataType == PIM_UINT16 || dataType == PIM_UINT32 || dataType == PIM_UINT64) {
      // Assume pop count reduction circut in tR runtime
      msRuntime = ((m_tR + m_tR) * bitsPerElement);
      msRuntime *= numPass;
      // reduction for all regions
      msRuntime += static_cast<double>(numRegions) / 3200000;
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
  {
    // read a row to walker, then reduce in serial
    double numberOfOperationPerElement = ((double)bitsPerElement / m_flucrumAluBitWidth);
    msRuntime = m_tR + (maxElementsPerRegion * m_fulcrumAluLatency * numberOfOperationPerElement * numPass);
    // reduction for all regions
    msRuntime += static_cast<double>(numCore) / 3200000;
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
    msRuntime = m_tR + (maxElementsPerRegion * m_blimpCoreLatency * numberOfOperationPerElement * numPass);
    // reduction for all regions
    msRuntime += static_cast<double>(numCore) / 3200000;
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
  PimDeviceEnum simTarget = pimSim::get()->getSimTarget();
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned maxElementsPerRegion = obj.getMaxElementsPerRegion();

  switch (simTarget) {
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
  {
    // assume taking 1 ALU latency to write an element
    double numberOfOperationPerElement = ((double)bitsPerElement / m_flucrumAluBitWidth);
    msRuntime = m_tW + m_fulcrumAluLatency * maxElementsPerRegion * numberOfOperationPerElement;
    msRuntime *= numPass;
    break;
  }
  case PIM_DEVICE_BANK_LEVEL:
  {
    // assume taking 1 ALU latency to write an element
    double numberOfOperationPerElement = ((double)bitsPerElement / m_blimpCoreBitWidth);
    msRuntime = m_tW + (m_blimpCoreLatency * maxElementsPerRegion * numberOfOperationPerElement);
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
  PimDeviceEnum simTarget = pimSim::get()->getSimTarget();
  double msRuntime = 0.0;
  unsigned numPass = obj.getMaxNumRegionsPerCore();
  unsigned bitsPerElement = obj.getBitsPerElement();
  unsigned numRegions = obj.getRegions().size();

  switch (simTarget) {
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

