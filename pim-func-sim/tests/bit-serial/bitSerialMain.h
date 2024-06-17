// Bit-Serial Performance Modeling - Main
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef BIT_SERIAL_MAIN_H
#define BIT_SERIAL_MAIN_H

//! @class  bitSerialMain
//! @brief  Bit-serial performance tests main entry
class bitSerialMain
{
public:
  bitSerialMain() {}
  ~bitSerialMain() {}

  bool runAllTests();

private:

  bool testBitsimd();
  bool testBitsimdAp();
  bool testSimdram();

};

#endif

