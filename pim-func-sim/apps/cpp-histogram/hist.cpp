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

// void histogram(int imgdata_bytes, const std::vector<int> &imgData, std::vector<int> &red, std::vector<int> &green, 
//                std::vector<int> &blue, const std::vector<int> &redMask, const std::vector<int> &greenMask, const std::vector<int> &blueMask) 
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
//   PimObjId redMaskObj = pimAllocAssociated(bitsPerElement, imgObj, PIM_INT32);
//   if (redMaskObj == -1)
//   {
//     std::cout << "Abort" << std::endl;
//     return;
//   }
//   PimObjId greenMaskObj = pimAllocAssociated(bitsPerElement, imgObj, PIM_INT32);
//   if (greenMaskObj == -1)
//   {
//     std::cout << "Abort" << std::endl;
//     return;
//   }
//   PimObjId blueMaskObj = pimAllocAssociated(bitsPerElement, imgObj, PIM_INT32);
//   if (blueMaskObj == -1)
//   {
//     std::cout << "Abort" << std::endl;
//     return;
//   }
//   PimObjId tempResultObj = pimAllocAssociated(bitsPerElement, imgObj, PIM_INT32);
//   if (tempResultObj == -1)
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

//   // Start Mask setup
//   status = pimCopyHostToDevice((void *) redMask.data(), redMaskObj);
//   if (status != PIM_OK)
//   {
//     std::cout << "Abort" << std::endl;
//     return;
//   }
//   status = pimCopyHostToDevice((void *) greenMask.data(), greenMaskObj);
//   if (status != PIM_OK)
//   {
//     std::cout << "Abort" << std::endl;
//     return;
//   }
//   status = pimCopyHostToDevice((void *) blueMask.data(), blueMaskObj);
//   if (status != PIM_OK)
//   {
//     std::cout << "Abort" << std::endl;
//     return;
//   }
//   // End Mask setup

//   for (int i = 0; i < 256; ++i) 
//   {
//     status = pimBroadcast(keyObj, i);
//     int temp;
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

//     status = pimAnd(keyObj, blueMaskObj, tempResultObj);
//     if (status != PIM_OK)
//     {
//       std::cout << "Abort" << std::endl;
//       return;
//     }

//     temp = blue[i];
//     status = pimRedSumRanged(tempResultObj, 0, imgdata_bytes, &blue[i]);
//     if (status != PIM_OK)
//     {
//       std::cout << "Abort" << std::endl;
//       return;
//     }
//     blue[i] += temp;

//     status = pimAnd(keyObj, greenMaskObj, tempResultObj);
//     if (status != PIM_OK)
//     {
//       std::cout << "Abort" << std::endl;
//       return;
//     }

//     temp = green[i];
//     status = pimRedSumRanged(tempResultObj, 0, imgdata_bytes, &green[i]);
//     if (status != PIM_OK)
//     {
//       std::cout << "Abort" << std::endl;
//       return;
//     }
//     green[i] += temp;

//     status = pimAnd(keyObj, redMaskObj, tempResultObj);
//     if (status != PIM_OK)
//     {
//       std::cout << "Abort" << std::endl;
//       return;
//     }

//     temp = red[i];
//     status = pimRedSumRanged(tempResultObj, 0, imgdata_bytes, &red[i]);
//     if (status != PIM_OK)
//     {
//       std::cout << "Abort" << std::endl;
//       return;
//     }
//     red[i] += temp;
//     std::cout << "Blue: '" << blue[i] << "' | Green: '" << green[i] << "' | Red: '" << red[i] << "'" << std::endl;
//   }
//   pimFree(imgObj);
//   pimFree(keyObj);
//   pimFree(blueMaskObj);
//   pimFree(greenMaskObj);
//   pimFree(redMaskObj);
//   pimFree(tempResultObj);
// }

void histogram(int imgdata_bytes, const std::vector<int> &imgData, std::vector<int> &red, std::vector<int> &green, 
               std::vector<int> &blue) 
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

  PimStatus status = pimCopyHostToDevice((void *) imgData.data(), imgObj);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

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

    temp = blue[i];
    status = pimRedSumRanged(keyObj, 0, imgdata_bytes / 3, &blue[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    blue[i] += temp;

    temp = green[i];
    status = pimRedSumRanged(keyObj, imgdata_bytes / 3, imgdata_bytes / 3 * 2, &green[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    green[i] += temp;

    temp = red[i];
    status = pimRedSumRanged(keyObj, imgdata_bytes / 3 * 2, imgdata_bytes, &red[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    red[i] += temp;
    std::cout << "Blue: '" << blue[i] << "' | Green: '" << green[i] << "' | Red: '" << red[i] << "'" << std::endl;
  }
  pimFree(imgObj);
  pimFree(keyObj);
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::string fn = params.inputFile;
  std::cout << "Input file name: '" << fn << "'\n";
  int fd;
  uint64_t imgdata_bytes;
  struct stat finfo;
  char* fdata;
  // unsigned short* bitsperpixel; 
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

  // bitsperpixel = (unsigned short *)(&(fdata[BITS_PER_PIXEL_POS]));
  // bitsperpixel = 24 by definition of a RGB .bmp file, unused as no need for endianess check 
  // (.bmp files are always little endian) and input is assumed as 24-bit .bmp file

  data_pos = (unsigned short *)(&(fdata[IMG_DATA_OFFSET_POS]));
  imgdata_bytes = static_cast<uint64_t> (finfo.st_size) - static_cast<uint64_t>(*(data_pos));
  printf("This file has %ld bytes of image data, %ld pixels\n", imgdata_bytes,
                                                            imgdata_bytes / 3);

  if (!createDevice(params.configFile))
    return 1;

  std::vector<int> red(256, 0), green(256, 0), blue(256, 0);

  std::vector<uint8_t> imgData(fdata + *data_pos, fdata + finfo.st_size);
  std::vector<int> imgDataToInt, orderedImgData;

  for (uint64_t i = 0; i < imgdata_bytes; ++i) {
    imgDataToInt.push_back(static_cast<int> (imgData[i]));
  }

  for (uint64_t i = 0; i < imgdata_bytes; i+=3) {
    orderedImgData.push_back(imgDataToInt[i]);
  }

  for (uint64_t i = 1; i < imgdata_bytes; i+=3) {
    orderedImgData.push_back(imgDataToInt[i]);
  }

  for (uint64_t i = 2; i < imgdata_bytes; i+=3) {
    orderedImgData.push_back(imgDataToInt[i]);
  }  

  uint64_t numCol = 8192, numRow = 8192, numCore = 4096;
  uint64_t totalAvailableBits = numCol * numRow * numCore;
  uint64_t requiredBitsforImage = (imgdata_bytes * 32) + 32;
  std::cout << "reqdBitsForImage : " << requiredBitsforImage << std::endl;
  int numItr = std::ceil(static_cast<double>(requiredBitsforImage) / totalAvailableBits);
  std::cout << "numItr : " << numItr << std::endl;

  if (numItr == 1)
  {
    histogram(imgdata_bytes, orderedImgData, red, green, blue);
  }
  else
  {
    // TODO
    uint64_t bytesPerChunk = totalAvailableBits / 8;

    for (int itr = 0; itr < numItr; ++itr)
    {
      uint64_t startByte = itr * bytesPerChunk;
      uint64_t endByte = std::min(startByte + bytesPerChunk, imgdata_bytes);
      uint64_t chunkSize = endByte - startByte;

      std::vector<int> imgDataChunk(imgDataToInt.begin() + startByte, imgDataToInt.begin() + endByte);
      histogram(chunkSize, imgDataChunk, red, green, blue);
    }
  }

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

    if (errorFlag)
      std::cout << "At least one wrong answer" << std::endl;
    else
      std::cout << "Correct!" << std::endl;
  }

  pimShowStats();

  munmap(fdata, finfo.st_size + 1);
  close(fd);

  return 0;
}
