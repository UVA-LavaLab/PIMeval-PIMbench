// Test: C++ version of Brightness
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <cassert>

#include "../../util.h"
#include "libpimeval.h"

using namespace std;

#define MINCOLORVALUE 0 // Sets the max value that any color channel can be in a given pixel
#define MAXCOLORVALUE 255 // Sets the max value that any color channel can be in a given pixel

// Params ---------------------------------------------------------------------
typedef struct Params
{
  char *configFile;
  std::string inputFile;
  bool shouldVerify;
  int brightnessCoefficient;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./brightness.out [options]"
          "\n"
          "\n    -c    dramsim config file"
          "\n    -i    24-bit .bmp input file (default=uses 'sample1.bmp' from '/cpp-histogram/histogram_datafiles/' directory)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n    -b    brightness coefficient value (default=20)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.configFile = nullptr;
  p.inputFile = "../../histogram/histogram_datafiles/sample1.bmp";
  p.shouldVerify = false;
  p.brightnessCoefficient = 20;

  int opt;
  while ((opt = getopt(argc, argv, "h:c:i:v:b:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
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
    case 'b':
      p.brightnessCoefficient = strtoull(optarg, NULL, 0);
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

void brightness(uint64_t imgDataBytes, const std::vector<int16_t> &imgData, int coefficient, std::vector<int16_t> &resultData)
{
  PimObjId imgObj = pimAlloc(PIM_ALLOC_AUTO, imgDataBytes, PIM_INT16);
  assert(imgObj != -1);
  PimObjId additionObj = pimAllocAssociated(imgObj, PIM_INT16);
  assert(additionObj != -1);
  PimObjId minObj = pimAllocAssociated(imgObj, PIM_INT16);
  assert(minObj != -1);
  PimObjId resultObj = pimAllocAssociated(imgObj, PIM_INT16);
  assert(resultObj != -1);

  PimStatus status = pimCopyHostToDevice((void *) imgData.data(), imgObj);
  assert(status == PIM_OK);

  status = pimAddScalar(imgObj, additionObj, coefficient);
  assert(status == PIM_OK);
  status = pimMinScalar(additionObj, minObj, MAXCOLORVALUE);
  assert(status == PIM_OK);
  status = pimMaxScalar(minObj, resultObj, MINCOLORVALUE);
  assert(status == PIM_OK);

  status = pimCopyDeviceToHost(resultObj, (void *) resultData.data());
  assert(status == PIM_OK);

  pimFree(imgObj);
  pimFree(additionObj);
  pimFree(minObj);
  pimFree(resultObj);
}

int truncate(int16_t pixelValue, int brightnessCoefficient)
{
  return std::min(MAXCOLORVALUE, std::max(MINCOLORVALUE, pixelValue + brightnessCoefficient));
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::string fn = params.inputFile;
  std::cout << "Running Image Brightness on PIM for input file : '" << fn << "'" << std::endl;

  int fd;
  uint64_t imgDataBytes;
  struct stat finfo;
  char* fdata;
  unsigned short* dataPos;
  int imgDataOffsetPosition;

  // Start data parsing
  if (fn.substr(fn.find_last_of(".")) != ".bmp")
  {
    // TODO: reading in other types of input files
    std::cout << "Need work reading in other file types" << std::endl;
    return 1;
  }
  else
  {
    // Assuming input will be 24-bit .bmp files
    fd = open(params.inputFile.c_str(), O_RDONLY);
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

    imgDataOffsetPosition = 10; // Start of image data, ignoring unneeded header data and info
    // Defined according to the assumed input file structure given

    dataPos = (unsigned short *)(&(fdata[imgDataOffsetPosition]));
    imgDataBytes = static_cast<uint64_t> (finfo.st_size) - static_cast<uint64_t>(*(dataPos));
  }
  // End data parsing

  printf("This file has %llu bytes of image data with a brightness coefficient of %d\n", imgDataBytes, params.brightnessCoefficient);

  if (!createDevice(params.configFile))
  {
    return 1;
  }

  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  assert(status == PIM_OK);

  std::vector<uint8_t> imgData(fdata + *dataPos, fdata + finfo.st_size);
  std::vector<int16_t> imgDataToInt16(imgDataBytes), resultData(imgDataBytes);
  // Converting to int16_t to prevent potential overflow situations when adding or subtracting the brightness coefficient
  //    exceeds limits for uint8_t

  for (uint64_t i = 0; i < imgDataBytes; ++i)
  {
    imgDataToInt16[i] = (static_cast<int16_t> (imgData[i]));
  }

  uint64_t numCol = deviceProp.numColPerSubarray, numRow = deviceProp.numRowPerSubarray, 
           numCore = deviceProp.numRanks * deviceProp.numBankPerRank * deviceProp.numSubarrayPerBank;
  uint64_t totalAvailableBits = numCol * numRow * numCore;
  uint64_t requiredBitsforImage = (imgDataBytes * 32) + 32;
  int numItr = std::ceil(static_cast<double> (requiredBitsforImage) / totalAvailableBits);
  std::cout << "Required iterations for image: " << numItr << std::endl;

  if (numItr == 1)
  {
    brightness(imgDataBytes, imgDataToInt16, params.brightnessCoefficient, resultData);
  }
  else
  {
    //TODO: ensure large inputs can be run in multiple brightness() calls if they can't fit in one PIM object
    uint64_t bytesPerChunk = totalAvailableBits / 8;
    std::vector<int16_t> tempResult;

    for (int itr = 0; itr < numItr; ++itr)
    {
      uint64_t startByte = itr * bytesPerChunk;
      uint64_t endByte = std::min(startByte + bytesPerChunk, imgDataBytes);
      uint64_t chunkSize = endByte - startByte;

      std::vector<int16_t> imgDataChunk(imgDataToInt16.begin() + startByte, imgDataToInt16.begin() + endByte);
      brightness(chunkSize, imgDataChunk, params.brightnessCoefficient, tempResult);
      resultData.insert(resultData.end(), tempResult.begin(), tempResult.end());
    }
  }

  if (params.shouldVerify)
  {  
    int errorFlag = 0;
    for (uint64_t i = 0; i < imgDataBytes; ++i) 
    {     
      imgDataToInt16[i] = truncate(imgDataToInt16[i], params.brightnessCoefficient); 
      if (imgDataToInt16[i] != resultData[i])
      {
        std::cout << "Wrong answer at index " << i << " | Wrong PIM answer = " << resultData[i] << " (CPU expected = " << imgDataToInt16[i] << ")" << std::endl;
        errorFlag = 1;
      }
    }
    if (!errorFlag)
    {
      std::cout << "Correct!" << std::endl;
    }
  }

  pimShowStats();

  munmap(fdata, finfo.st_size + 1);
  close(fd);

  return 0;
}
