// Test: C++ version of histogram
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cmath>
#include <cstring>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../util.h"
#include "libpimsim.h"

#define IMG_DATA_OFFSET_POS 10
#define BITS_PER_PIXEL_POS 28

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t dataSize;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./lr [options]"
          "\n"
          "\n    -l    input size (default=65536 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing 2D matrix (default=generates matrix with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.dataSize = NULL;
  p.configFile = nullptr;
  p.inputFile = "sample1.bmp";
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:l:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'l':
      p.dataSize = strtoull(optarg, NULL, 0);
      break;
    case 'c':
      p.configFile = optarg;
      break;
    case 'i':
      p.inputFile = optarg;
      break;
    case 'v':
      p.shouldVerify = (*optarg == 't') ? true : false;
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

void histogram(int imgdata_bytes, const std::vector<int> &imgData, std::vector<int> &red, std::vector<int> &green, std::vector<int> &blue) 
{
  unsigned bitsPerElement = sizeof(int) * 8;
  PimObjId imgObj = pimAlloc(PIM_ALLOC_AUTO, imgdata_bytes, bitsPerElement, PIM_INT32);
  if (imgObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  PimObjId keyObj = pimAllocAssociated(bitsPerElement, imgObj, PIM_INT32);
  if (keyObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  PimObjId redMaskObj = pimAllocAssociated(bitsPerElement, imgObj, PIM_INT32);
  if (redMaskObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  PimObjId greenMaskObj = pimAllocAssociated(bitsPerElement, imgObj, PIM_INT32);
  if (greenMaskObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  PimObjId blueMaskObj = pimAllocAssociated(bitsPerElement, imgObj, PIM_INT32);
  if (blueMaskObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  PimObjId tempResultObj = pimAllocAssociated(bitsPerElement, imgObj, PIM_INT32);
  if (tempResultObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice((void *) imgData.data(), imgObj);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  // Start Mask setup
  std::vector<int> redMask(imgdata_bytes, 0), greenMask(imgdata_bytes, 0), blueMask(imgdata_bytes, 0);
  for (int i = 0; i < imgdata_bytes; i+=3) 
  {
    redMask[i + 2] = 1;
    greenMask[i + 1] = 1;
    blueMask[i] = 1;
  }
  status = pimCopyHostToDevice((void *) redMask.data(), redMaskObj);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  status = pimCopyHostToDevice((void *) greenMask.data(), greenMaskObj);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  status = pimCopyHostToDevice((void *) blueMask.data(), blueMaskObj);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  // End Mask setup
  std::vector<int> temp(imgdata_bytes, 0);

  for (int i = 0; i < 256; ++i) 
  {
    status = pimBroadcast(keyObj, i);
    int temp;
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimEQ(keyObj, imgObj, keyObj);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    // status = pimCopyDeviceToHost(keyObj, (void *) temp.data());
    // for (int i = 0; i < imgdata_bytes; ++i) {
    //   std::cout << temp[i] << std::endl;
    // }

    status = pimAnd(keyObj, blueMaskObj, tempResultObj);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    // status = pimPopCount(tempResultObj, tempResultObj);
    // if (status != PIM_OK)
    // {
    //   std::cout << "Abort" << std::endl;
    //   return;  
    // }
    // status = pimCopyDeviceToHost(tempResultObj, (void *) temp.data());
    // blue[i] = temp[0];
    temp = blue[i];
    status = pimRedSumRanged(tempResultObj, 0, imgdata_bytes, &blue[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    blue[i] += temp;

    status = pimAnd(keyObj, greenMaskObj, tempResultObj);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    // status = pimPopCount(tempResultObj, tempResultObj);
    // if (status != PIM_OK)
    // {
    //   std::cout << "Abort" << std::endl;
    //   return;  
    // }
    // status = pimCopyDeviceToHost(tempResultObj, (void *) temp.data());
    // green[i] = temp[0];
    temp = green[i];
    status = pimRedSumRanged(tempResultObj, 0, imgdata_bytes, &green[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    green[i] += temp;

    status = pimAnd(keyObj, redMaskObj, tempResultObj);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    // status = pimPopCount(tempResultObj, tempResultObj);
    // if (status != PIM_OK)
    // {
    //   std::cout << "Abort" << std::endl;
    //   return;  
    // }
    // status = pimCopyDeviceToHost(tempResultObj, (void *) temp.data());
    // blue[i] = temp[0];
    temp = red[i];
    status = pimRedSumRanged(tempResultObj, 0, imgdata_bytes, &red[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    red[i] += temp;
    std::cout << "Blue: '" << blue[i] << "' | Green: '" << green[i] << "' | Red: '" << red[i] << "'" << std::endl;
    //temp.clear();
  }
  pimFree(imgObj);
  pimFree(keyObj);
  pimFree(blueMaskObj);
  pimFree(greenMaskObj);
  pimFree(redMaskObj);
  pimFree(tempResultObj);
}

// void histogram(int imgdata_bytes, const std::vector<int> &imgData, std::vector<int> &counter) 
// {
//   unsigned bitsPerElement = sizeof(int) * 8;
//   PimObjId imgObj = pimAlloc(PIM_ALLOC_AUTO, imgdata_bytes, bitsPerElement, PIM_INT32);
//   if (imgObj == -1)
//   {
//     std::cout << "Abort" << std::endl;
//     return;
//   }
//   PimObjId keyObj = pimAllocAssociated(bitsPerElement, imgObj, PIM_INT32);
//   if (keyObj == -1)
//   {
//     std::cout << "Abort" << std::endl;
//     return;
//   }

//   PimStatus status = pimCopyHostToDevice((void *) imgData.data(), imgObj);
//   if (status != PIM_OK)
//   {
//     std::cout << "Abort" << std::endl;
//     return;
//   }

//   std::vector<int> temp(imgdata_bytes);

//   for (int i = 0; i < 256; ++i) 
//   {
//     status = pimBroadcast(keyObj, i);
//     if (status != PIM_OK)
//     {
//       std::cout << "Abort" << std::endl;
//       return;
//     }
//     status = pimEQ(keyObj, imgObj, keyObj);
//     if (status != PIM_OK)
//     {
//       std::cout << "Abort" << std::endl;
//       return;
//     }
//     // status = pimPopCount(tempResultObj, tempResultObj);
//     // if (status != PIM_OK)
//     // {
//     //   std::cout << "Abort" << std::endl;
//     //   return;  
//     // }
//     // status = pimCopyDeviceToHost(tempResultObj, (void *) temp.data());
//     // blue[i] = temp[0];
//     status = pimRedSum(keyObj, &counter[i]);
//     if (status != PIM_OK)
//     {
//       std::cout << "Abort" << std::endl;
//       return;
//     }
//     std::cout << "i value: '" << i << "' = " << counter[i] << std::endl;
//   }
//   pimFree(imgObj);
//   pimFree(keyObj);
// }

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::string fn = params.inputFile;
  std::cout << "Input file name: '" << fn << "'\n";
  int fd, imgdata_bytes;
  struct stat finfo;
  char* fdata;
  //unsigned short* bitsperpixel; 
  unsigned short* data_pos;
  if (!fn.substr(fn.find_last_of(".") + 1).compare("bmp") == 0)
  {
    // TODO
    // assuming inputs will always be 24-bit .bmp files
    std::cout << "Need work reading in other file types" << std::endl;
    return 1;
  }

  fd = open(params.inputFile, O_RDONLY);
  if (fd < 0) 
  {
    perror("Failed to open input file, or file doesn't exist");
    return 1;
  }
  
  if (fstat(fd, &finfo) < 0) 
  {
    perror("Failed to get file info");
    return 1;
  }
  fdata = static_cast<char *>(mmap(0, finfo.st_size + 1, PROT_READ, MAP_PRIVATE, fd, 0));
  if (fdata == 0) 
  {
    perror("Failed to memory map the file");
    return 1;
  }

  //bitsperpixel = (unsigned short *)(&(fdata[BITS_PER_PIXEL_POS]));
  // bitsperpixel = 24 by definition of a RGB .bmp file, unused as no need for endianess check 
  // (.bmp files are always little endian) and input is assumed as 24-bit .bmp file

  data_pos = (unsigned short *)(&(fdata[IMG_DATA_OFFSET_POS]));
  imgdata_bytes = (int)finfo.st_size - (int)(*(data_pos));
  printf("This file has %d bytes of image data, %d pixels\n", imgdata_bytes,
                                                            imgdata_bytes / 3);

  if (!createDevice(params.configFile))
    return 1;

  std::vector<int> red(256, 0), green(256, 0), blue(256, 0);

  std::vector<uint8_t> imgData(fdata + *data_pos, fdata + finfo.st_size);
  std::vector<int> imgDataToInt;

  for (int i = 0; i < imgdata_bytes; ++i) {
    imgDataToInt.push_back(static_cast<int> (imgData[i]));
  }

  // std::vector<int> blueImgData;
  // std::vector<int> greenImgData;
  // std::vector<int> redImgData;

  // for (int i = 0; i < imgdata_bytes; i+=3) {
  //   // imgDataToInt.push_back(static_cast<int>(val));
  //   blueImgData.push_back(static_cast<int> (imgData[i]));
  //   greenImgData.push_back(static_cast<int> (imgData[i+1]));
  //   redImgData.push_back(static_cast<int> (imgData[i+2]));
  // }

  uint64_t numCol = 8192, numRow = 8192, numCore = 4096;
  uint64_t totalAvailableBits = numCol * numRow * numCore;
  uint64_t requiredBitsforImage = (imgdata_bytes * 32) + 32;
  int numItr = std::ceil(static_cast<double>(requiredBitsforImage) / totalAvailableBits);

  if (numItr == 1)
  {
    histogram(imgdata_bytes, imgDataToInt, red, green, blue);
  }
  else
  {
    // TODO
    uint64_t bitsPerChunk = totalAvailableBits - 32;
    uint64_t bytesPerChunk = bitsPerChunk / 32;
    for (int itr = 0; itr < numItr; ++itr)
    {
      uint64_t startByte = itr * totalAvailableBits;
      uint64_t endByte = std::min(startByte + (totalAvailableBits * 32), static_cast<uint64_t>(imgdata_bytes));
      uint64_t chunkSize = endByte - startByte;

      std::vector<int> imgDataChunk(imgDataToInt.begin() + startByte, imgDataToInt.begin() + endByte);
      histogram(chunkSize, imgDataChunk, red, green, blue);
    }
  }

  // histogram(imgdata_bytes/3, blueImgData, blue);
  // histogram(imgdata_bytes/3, greenImgData, green);
  // histogram(imgdata_bytes/3, redImgData, red);

  // std::vector<int> slicing;
  // for (int i = 0; i < imgdata_bytes; ++i) {
  //   if (i % (8192 * 8192) == 0) {
  //     histogram(8192 * 8192, slicing, red, green, blue);
  //     slicing.clear();
  //   }
  //   slicing.push_back(imgData[i]);
  // }

  // histogram(imgdata_bytes, imgDataToInt, red, green, blue);

  if (params.shouldVerify)
  {
    int red_cpu[256];
    int green_cpu[256];
    int blue_cpu[256];

    memset(&(red_cpu[0]), 0, sizeof(int) * 256);
    memset(&(green_cpu[0]), 0, sizeof(int) * 256);
    memset(&(blue_cpu[0]), 0, sizeof(int) * 256);
   
    for (int i=*data_pos; i < finfo.st_size; i+=3) {      
      unsigned char *val = (unsigned char *)&(fdata[i]);
      blue_cpu[*val]++;
      
      val = (unsigned char *)&(fdata[i+1]);
      green_cpu[*val]++;
      
      val = (unsigned char *)&(fdata[i+2]);
      red_cpu[*val]++;   
    }

    int errorFlag = 0;
    for (int i = 0; i < 256; ++i)
    {
      if (red_cpu[i] != red[i]) {
        std::cout << "Wrong answer for red: " << red[i] << " | " << i << " (expected " << red_cpu[i] << ")" << std::endl;
        errorFlag = 1;
      }
      if (green_cpu[i] != green[i]) {
        std::cout << "Wrong answer for green: " << green[i] << " | " << i << " (expected " << green_cpu[i] << ")" << std::endl;
        errorFlag = 1;
      }
      if (blue_cpu[i] != blue[i]) {
        std::cout << "Wrong answer for blue: " << blue[i] << " | " << i << " (expected " << blue_cpu[i] << ")" << std::endl;
        errorFlag = 1;
      }
    }

    if (errorFlag == 1)
        std::cout << "At least one wrong answer" << std::endl;
    else
        std::cout << "Correct!" << std::endl;
  }

  pimShowStats();

  munmap(fdata, finfo.st_size + 1);
  close(fd);

  return 0;
}
