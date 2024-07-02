// Test: C++ version of Histogram
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

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
#include "libpimsim.h"

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t dataSize;
  char *configFile;
  std::string inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./lr [options]"
          "\n"
          "\n    -l    input size (default=65536 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    24-bit .bmp input file (default=uses 'small.bmp' from 'histogram_datafiles' directory)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.dataSize = 0; 
  p.configFile = nullptr;
  p.inputFile = "histogram_datafiles/small.bmp";
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

void orderData(std::vector<uint8_t> &orderedImgData, const std::vector<uint8_t> imgData, uint64_t idxBegin, uint64_t idxEnd, int numChannels)
{
  // Splitting apart individual pixel values by grouping image data vector by color channels, going blue, green, then red
  for (int i = 0; i < numChannels; ++i)
  {
    for (uint64_t j = idxBegin; j < idxEnd; j+=3)
    {
      orderedImgData.push_back(imgData[j + i]);
    }
  }
}

void histogram(uint64_t imgDataBytes, const std::vector<uint8_t> &imgData, int numBins, 
               std::vector<uint64_t> &redCount, std::vector<uint64_t> &greenCount, std::vector<uint64_t> &blueCount) 
{
  unsigned bitsPerElement = sizeof(uint8_t) * 8;
  PimObjId imgObj = pimAlloc(PIM_ALLOC_AUTO, imgDataBytes, bitsPerElement, PIM_UINT8);
  assert(imgObj != -1);
  PimObjId tempObj = pimAllocAssociated(bitsPerElement, imgObj, PIM_UINT8);
  assert(tempObj != -1);

  PimStatus status = pimCopyHostToDevice((void *) imgData.data(), imgObj);
  assert(status == PIM_OK);

  uint64_t prevCount; // Needed when image data can't fit in one PIM object, multiple passes are needed

  for (int i = 0; i < numBins; ++i) 
  {
    status = pimEQScalar(imgObj, tempObj, static_cast<uint64_t> (i));
    assert(status == PIM_OK);

    prevCount = blueCount[i];
    status = pimRedSumRangedUInt(tempObj, 0, imgDataBytes / 3, &blueCount[i]);
    assert(status == PIM_OK);
    blueCount[i] += prevCount;

    prevCount = greenCount[i];
    status = pimRedSumRangedUInt(tempObj, imgDataBytes / 3, imgDataBytes / 3 * 2, &greenCount[i]);
    assert(status == PIM_OK);
    greenCount[i] += prevCount;

    prevCount = redCount[i];
    status = pimRedSumRangedUInt(tempObj, imgDataBytes / 3 * 2, imgDataBytes, &redCount[i]);
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
  int numChannels, imgDataOffsetPosition, numBins;
  struct stat finfo;
  char* fdata;
  unsigned short* dataPos;

  // Start data parsing
  if (!fn.substr(fn.find_last_of(".") + 1).compare("bmp") == 0)
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

    numBins = 256; // RGB values at any given pixel can be a value 0 to 255 (inclusive)
    numChannels = 3; // Red, green, and blue color channels
    imgDataOffsetPosition = 10; // Start of image data, ignoring unneeded header data and info
    // Defined according to the assumed input file structure given

    dataPos = (unsigned short *)(&(fdata[imgDataOffsetPosition]));
    imgDataBytes = static_cast<uint64_t> (finfo.st_size) - static_cast<uint64_t>(*(dataPos));
  }
  // End data parsing

  printf("This file has %ld bytes of image data, %ld pixels\n", imgDataBytes, imgDataBytes / numChannels);

  if (!createDevice(params.configFile))
  {
    return 1;
  }

  std::vector<uint64_t> redCount(numBins, 0), greenCount(numBins, 0), blueCount(numBins, 0);
  std::vector<uint8_t> imgData(fdata + *dataPos, fdata + finfo.st_size);
  std::vector<uint8_t> orderedImgData;

  uint64_t numCol = 8192, numRow = 8192, numCore = 4096;
  uint64_t totalAvailableBits = numCol * numRow * numCore;
  uint64_t requiredBitsforImage = ((imgDataBytes * 32) + 32);
  int numItr = std::ceil(static_cast<double> (requiredBitsforImage) / totalAvailableBits);
  std::cout << "Required iterations for image: " << numItr << std::endl;

  if (numItr == 1)
  {
    orderData(orderedImgData, imgData, 0, imgDataBytes, numChannels);
    histogram(imgDataBytes, orderedImgData, numBins, redCount, greenCount, blueCount);
  }
  else
  {
    //TODO: ensure large inputs can be run in multiple histogram() calls if they can't fit in one PIM object
    uint64_t bytesPerChunk = totalAvailableBits / 8;

    for (int itr = 0; itr < numItr; ++itr)
    {
      uint64_t startByte = itr * bytesPerChunk;
      uint64_t endByte = std::min(startByte + bytesPerChunk, imgDataBytes);
      uint64_t chunkSize = endByte - startByte;

      std::vector<uint8_t> imgDataChunk(imgData.begin() + startByte, imgData.begin() + endByte);
      orderData(orderedImgData, imgDataChunk, startByte, endByte, numChannels);

      histogram(chunkSize, orderedImgData, numBins, redCount, greenCount, blueCount);

      orderedImgData.clear();
    }
  }

  if (params.shouldVerify)
  {
    uint64_t redCheck[numBins];
    uint64_t greenCheck[numBins];
    uint64_t blueCheck[numBins];
    int errorFlag = 0;

    memset(&(redCheck[0]), 0, sizeof(uint64_t) * numBins);
    memset(&(greenCheck[0]), 0, sizeof(uint64_t) * numBins);
    memset(&(blueCheck[0]), 0, sizeof(uint64_t) * numBins);
   
    for (int i=*dataPos; i < finfo.st_size; i+=3) 
    {      
      unsigned char *val = (unsigned char *)&(fdata[i]);
      blueCheck[*val]++;
      
      val = (unsigned char *)&(fdata[i+1]);
      greenCheck[*val]++;
      
      val = (unsigned char *)&(fdata[i+2]);
      redCheck[*val]++;   
    }

    for (int i = 0; i < numBins; ++i)
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
