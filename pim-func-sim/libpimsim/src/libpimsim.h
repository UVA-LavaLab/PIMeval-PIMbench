// File: libpimsim.h
// PIM Functional Simulator Library Interface
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#ifndef LAVA_LIB_PIM_SIM_H
#define LAVA_LIB_PIM_SIM_H

#ifdef __cplusplus
extern "C" {
#endif

  //! @brief  PIM API return status
  enum PimStatus {
    PIM_ERROR = 0,
    PIM_OK,
  };

  //! @brief  PIM device types
  enum PimDeviceEnum {
    PIM_DEVICE_NONE = 0,
    PIM_FUNCTIONAL,
  };

  //! @brief  PIM allocation types
  enum PimAllocEnum {
    PIM_ALLOC_V1 = 0,  // vertical layout, at most one region per core
    PIM_ALLOC_H1,      // horizontal layout, at most one region per core
  };

  //! @brief  PIM data copy types
  enum PimCopyEnum {
    PIM_COPY_V = 0,
    PIM_COPY_H,
  };

  typedef int PimCoreId;
  typedef int PimObjId;

  // Device creation and deletion
  PimStatus pimCreateDevice(PimDeviceEnum deviceType, unsigned numCores, unsigned numRows, unsigned numCols);
  PimStatus pimCreateDeviceFromConfig(PimDeviceEnum deviceType, const char* configFileName);
  PimStatus pimDeleteDevice();

  // Resource allocation and deletion
  PimObjId pimAlloc(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement);
  PimObjId pimAllocAssociated(PimAllocEnum allocType, unsigned numElements, unsigned bitsPerElement, PimObjId ref);
  PimStatus pimFree(PimObjId obj);

  // Data transfer
  PimStatus pimCopyHostToDevice(PimCopyEnum copyType, void* src, PimObjId dest);
  PimStatus pimCopyDeviceToHost(PimCopyEnum copyType, PimObjId src, void* dest);

  // Computation
  PimStatus pimAddInt32V(PimObjId src1, PimObjId src2, PimObjId dest);
  PimStatus pimRedSum(PimObjId a, PimObjId b, PimObjId c);


#ifdef __cplusplus
}
#endif

#endif

