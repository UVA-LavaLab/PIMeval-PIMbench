// Test: C++ version of vector add
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include "libpimsim.h"
#include "cstdio"

int main()
{
  std::printf("PIM test: Vector add\n");

  PimStatus status = pimCreateDevice(PIM_FUNCTIONAL, 4, 128, 256);
  if (status != PIM_OK) {
    std::printf("Abort\n");
    return 1;
  }

  PimObjId obj1 = pimAlloc(PIM_ALLOC_V1, 512, 32);
  if (obj1 == -1) {
    std::printf("Abort\n");
    return 1;
  }
  PimObjId obj2 = pimAllocAssociated(PIM_ALLOC_V1, 512, 32, obj1);
  if (obj2 == -1) {
    std::printf("Abort\n");
    return 1;
  }
  
  return 0;
}
