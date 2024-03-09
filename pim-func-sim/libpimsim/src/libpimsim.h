// File: libpimsim.h
// PIM Functional Simulator Library Interface
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_LIB_PIM_SIM_H
#define LAVA_LIB_PIM_SIM_H

#ifdef __cplusplus
extern "C" {
#endif

  enum PimStatus {
    PIM_ERROR = 0,
    PIM_OK,
  };

  enum PimDeviceEnum {
    PIM_DEVICE_NONE = 0,
    PIM_FUNCTIONAL,
  };

  enum PimAllocEnum {
    PIM_ALLOC_V = 0,
    PIM_ALLOC_H,
  };

  enum PimCopyEnum {
    PIM_COPY_V = 0,
    PIM_COPY_H,
  };

  typedef int PimObjId;

  // Device creation and deletion
  PimStatus pimCreateDevice(PimDeviceEnum deviceType, int numCores, int numRows, int numCols);
  PimStatus pimCreateDeviceFromConfig(PimDeviceEnum deviceType, const char* configFileName);
  PimStatus pimDeleteDevice();

  // Resource allocation and deletion
  PimObjId pimAlloc(PimAllocEnum allocType, int numElements, int bitsPerElement);
  PimObjId pimAllocAssociated(PimAllocEnum allocType, int numElements, int bitsPerElement, PimObjId ref);
  PimStatus pimFree(PimObjId obj);

  // Data transfer
  PimStatus pimCopyHostToDevice(PimCopyEnum copyType, void* src, PimObjId dest);
  PimStatus pimCopyDeviceToHost(PimCopyEnum copyType, PimObjId src, void* dest);

  // Computation
  PimStatus pimAdd(PimObjId src1, PimObjId src2, PimObjId dest);
  PimStatus pimRedSum(PimObjId a, PimObjId b, PimObjId c);


#ifdef __cplusplus
}
#endif

#endif

