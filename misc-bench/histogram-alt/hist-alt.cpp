// Test: C++ version of Histogram alternative
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

#include "../util.h"
#include "libpimeval.h"

#define NUMBINS 256 // RGB values at any given pixel can be a value 0 to 255 (inclusive)
#define NUMCHANNELS 3 // Red, green, and blue color channels

// Params ---------------------------------------------------------------------
typedef struct Params
{
  char *configFile;
  std::string inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./hist-alt.out [options]"
          "\n"
          "\n    -c    dramsim config file"
          "\n    -i    24-bit .bmp input file (default=uses 'sample1.bmp' from 'histogram_datafiles' directory)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.configFile = nullptr;
  p.inputFile = "../../PIMbench/histogram/histogram_datafiles/sample1.bmp";
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:c:i:v:")) >= 0)
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
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

void histogram(uint64_t imgDataBytes, const std::vector<uint8_t> &imgData, std::vector<uint64_t> &redCount, std::vector<uint64_t> &greenCount, std::vector<uint64_t> &blueCount) 
{
  PimObjId imgObj = pimAlloc(PIM_ALLOC_AUTO, imgDataBytes, PIM_UINT8);
  assert(imgObj != -1);
  PimObjId tempObj = pimAllocAssociated(imgObj, PIM_UINT8);
  assert(tempObj != -1);

  PimStatus status = pimCopyHostToDevice((void *) imgData.data(), imgObj);
  assert(status == PIM_OK);

  uint64_t prevCount; // Needed when image data can't fit in one PIM object, multiple passes are needed

  for (int i = 0; i < NUMBINS; ++i) 
  {
    status = pimEQScalar(imgObj, tempObj, static_cast<uint64_t> (i));
    assert(status == PIM_OK);

    prevCount = blueCount[i];
    status = pimRedSumRangedUInt(tempObj, 0, imgDataBytes / 3, &blueCount[i]);
    assert(status == PIM_OK);
    blueCount[i] += prevCount;

    prevCount = greenCount[i];
    status = pimRedSumRangedUInt(tempObj, imgDataBytes / 3, imgDataBytes * 2 / 3, &greenCount[i]);
    assert(status == PIM_OK);
    greenCount[i] += prevCount;

    prevCount = redCount[i];
    status = pimRedSumRangedUInt(tempObj, imgDataBytes * 2 / 3, imgDataBytes, &redCount[i]);
    assert(status == PIM_OK);
    redCount[i] += prevCount;
  }
  pimFree(imgObj);
  pimFree(tempObj);
}


int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::string fn = params.inputFile;
  std::cout << "Input file : '" << fn << "'" << std::endl;
  int fd;
  uint64_t imgDataBytes;
  int imgDataOffsetPosition;
  struct stat finfo;
  char* fdata;
  unsigned short* dataPos;

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

  printf("This file has %lu bytes of image data, %lu pixels\n", imgDataBytes, imgDataBytes / NUMCHANNELS);

  if (!createDevice(params.configFile))
  {
    return 1;
  }

  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  assert(status == PIM_OK);

  std::vector<uint64_t> redCount(NUMBINS, 0), greenCount(NUMBINS, 0), blueCount(NUMBINS, 0);
  std::vector<uint8_t> imgData(fdata + *dataPos, fdata + finfo.st_size);

  uint64_t numCol = deviceProp.numColPerSubarray, numRow = deviceProp.numRowPerSubarray, 
           numCore = deviceProp.numRanks * deviceProp.numBankPerRank * deviceProp.numSubarrayPerBank;
  uint64_t totalAvailableBits = numCol * numRow * numCore;
  uint64_t requiredBitsforImage = ((imgDataBytes / NUMCHANNELS * 8) + 8); // Using uint8_t instead of int, only require 8 bits
  int numItr = std::ceil(static_cast<double> (requiredBitsforImage) / totalAvailableBits);

  // Ensure large inputs can be run in multiple histogram() calls if they can't fit in one PIM object
  uint64_t bytesPerChunk = totalAvailableBits / 8;
  std::vector<uint8_t> redData, greenData, blueData;

  for (uint64_t i = 0; i < imgDataBytes; i+=NUMCHANNELS)
  {
    blueData.push_back(imgData[i]);
    greenData.push_back(imgData[i + 1]);
    redData.push_back(imgData[i + 2]);
  }

  for (int itr = 0; itr < numItr; ++itr)
  {
    uint64_t startByte = itr * bytesPerChunk;
    uint64_t endByte = std::min(startByte + bytesPerChunk, imgDataBytes / NUMCHANNELS);
    uint64_t chunkSize = endByte - startByte;

    std::vector<uint8_t> redDataChunk(redData.begin() + startByte, redData.begin() + endByte);
    std::vector<uint8_t> greenDataChunk(greenData.begin() + startByte, greenData.begin() + endByte);
    std::vector<uint8_t> blueDataChunk(blueData.begin() + startByte, blueData.begin() + endByte);

    std::vector<uint8_t> orderedImgDataChunk;
    orderedImgDataChunk.reserve(redDataChunk.size() + greenDataChunk.size() + blueDataChunk.size());
    orderedImgDataChunk.insert(orderedImgDataChunk.end(), blueDataChunk.begin(), blueDataChunk.end());
    orderedImgDataChunk.insert(orderedImgDataChunk.end(), greenDataChunk.begin(), greenDataChunk.end());
    orderedImgDataChunk.insert(orderedImgDataChunk.end(), redDataChunk.begin(), redDataChunk.end());

    histogram(chunkSize * NUMCHANNELS, orderedImgDataChunk, redCount, greenCount, blueCount);
  }

  if (params.shouldVerify)
  {
    uint64_t redCheck[NUMBINS];
    uint64_t greenCheck[NUMBINS];
    uint64_t blueCheck[NUMBINS];
    int errorFlag = 0;

    memset(&(redCheck[0]), 0, sizeof(uint64_t) * NUMBINS);
    memset(&(greenCheck[0]), 0, sizeof(uint64_t) * NUMBINS);
    memset(&(blueCheck[0]), 0, sizeof(uint64_t) * NUMBINS);
   
    for (int i=*dataPos; i < finfo.st_size; i+=NUMCHANNELS) 
    {      
      unsigned char *val = (unsigned char *)&(fdata[i]);
      blueCheck[*val]++;
      
      val = (unsigned char *)&(fdata[i+1]);
      greenCheck[*val]++;
      
      val = (unsigned char *)&(fdata[i+2]);
      redCheck[*val]++;   
    }

    for (int i = 0; i < NUMBINS; ++i)
    {
      if (redCheck[i] != redCount[i]) 
      {
        std::cout << "Index " << i << " | Wrong PIM answer for red = " << redCount[i] << " (CPU expected = " << redCheck[i] << ")" << std::endl;
        errorFlag = 1;
      }
      if (greenCheck[i] != greenCount[i]) 
      {
        std::cout << "Index " << i << " | Wrong PIM answer for green = " << greenCount[i] << " (CPU expected = " << greenCheck[i] << ")" << std::endl;
        errorFlag = 1;
      }
      if (blueCheck[i] != blueCount[i]) 
      {
        std::cout << "Index " << i << " | Wrong PIM answer for blue = " << blueCount[i] << " (CPU expected = " << blueCheck[i] << ")" << std::endl;
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
