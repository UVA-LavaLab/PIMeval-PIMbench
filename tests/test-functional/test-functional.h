// Test: Test functional behavior
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef PIMEVAL_TESTS_FUNCTIONAL_H
#define PIMEVAL_TESTS_FUNCTIONAL_H

//! @class  testFunctional
//! @brief  Functional test for high-level PIM APIs and data types
class testFunctional {
public:
   bool runAllTests();

private:
   void testI8();
   void testI16();
   void testI32();
   void testI64();
   void testU8();
   void testU16();
   void testU32();
   void testU64();

   void createPimDevice();
   void deletePimDevice();
};

#endif

