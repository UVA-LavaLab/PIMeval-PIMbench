// Test: C++ version of Histogram
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
#include <cstring>
#include <cassert>
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

void orderData(std::vector<uint8_t> &orderedImgData, const std::vector<uint8_t> imgData, uint64_t idxBegin, uint64_t idxEnd)
{
  for (int i = 0; i < 3; ++i)
  {
    for (uint64_t j = idxBegin; j < idxEnd; j+=3)
    {
      orderedImgData.push_back(imgData[j + i]);
    }
  }
}

void histogram(uint64_t imgdata_bytes, const std::vector<uint8_t> &imgData, std::vector<uint64_t> &redCount, std::vector<uint64_t> &greenCount, std::vector<uint64_t> &blueCount) 
{
  unsigned bitsPerElement = sizeof(uint8_t) * 8;
  PimObjId imgObj = pimAlloc(PIM_ALLOC_AUTO, imgdata_bytes, bitsPerElement, PIM_UINT8);
  assert(imgObj != -1);
  PimObjId tempObj = pimAllocAssociated(bitsPerElement, imgObj, PIM_UINT8);
  assert(tempObj != -1);

  PimStatus status = pimCopyHostToDevice((void *) imgData.data(), imgObj);
  assert(status == PIM_OK);

  uint64_t temp;

  for (uint64_t i = 0; i < 256; ++i) 
  {
    status = pimEQScalar(imgObj, tempObj, i);
    assert(status == PIM_OK);

    temp = blueCount[i];
    status = pimRedSumRangedUInt(tempObj, 0, imgdata_bytes / 3, &blueCount[i]);
    assert(status == PIM_OK);
    blueCount[i] += temp;

    temp = greenCount[i];
    status = pimRedSumRangedUInt(tempObj, imgdata_bytes / 3, imgdata_bytes / 3 * 2, &greenCount[i]);
    assert(status == PIM_OK);
    greenCount[i] += temp;

    temp = redCount[i];
    status = pimRedSumRangedUInt(tempObj, imgdata_bytes / 3 * 2, imgdata_bytes, &redCount[i]);
    assert(status == PIM_OK);
    redCount[i] += temp;
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
  uint64_t imgdata_bytes;
  struct stat finfo;
  char* fdata;
  unsigned short* data_pos;

  // Start data parsing
  if (!fn.substr(fn.find_last_of(".") + 1).compare("bmp") == 0)
  {
    // TODO: Assuming inputs will be 24-bit .bmp files
    std::cout << "Need work reading in other file types" << std::endl;
    return 1;
  }
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

  data_pos = (unsigned short *)(&(fdata[IMG_DATA_OFFSET_POS]));
  imgdata_bytes = static_cast<uint64_t> (finfo.st_size) - static_cast<uint64_t>(*(data_pos));
  printf("This file has %ld bytes of image data, %ld pixels\n", imgdata_bytes, imgdata_bytes / 3);
  // End data parsing

  if (!createDevice(params.configFile))
  {
    return 1;
  }

  std::vector<uint64_t> redCount(256, 0), greenCount(256, 0), blueCount(256, 0);
  std::vector<uint8_t> imgData(fdata + *data_pos, fdata + finfo.st_size);

  std::vector<uint8_t> orderedImgData;

  uint64_t numCol = 8192, numRow = 8192, numCore = 4096;
  uint64_t totalAvailableBits = numCol * numRow * numCore;
  uint64_t requiredBitsforImage = ((imgdata_bytes * 32) + 32);
  int numItr = std::ceil(static_cast<double> (requiredBitsforImage) / totalAvailableBits);
  std::cout << "Required iterations for image: " << numItr << std::endl;

  if (numItr == 1)
  {
    orderData(orderedImgData, imgData, 0, imgdata_bytes);
    histogram(imgdata_bytes, orderedImgData, redCount, greenCount, blueCount);
  }
  else
  {
    //TODO: ensure large inputs can be run in multiple histogram() calls if they can't fit in one PIM object
    uint64_t bytesPerChunk = totalAvailableBits / 8;

    for (int itr = 0; itr < numItr; ++itr)
    {
      uint64_t startByte = itr * bytesPerChunk;
      uint64_t endByte = std::min(startByte + bytesPerChunk, imgdata_bytes);
      uint64_t chunkSize = endByte - startByte;

      std::vector<uint8_t> imgDataChunk(imgData.begin() + startByte, imgData.begin() + endByte);
      orderData(orderedImgData, imgDataChunk, startByte, endByte);

      histogram(chunkSize, orderedImgData, redCount, greenCount, blueCount);
    }
  }

  if (params.shouldVerify)
  {
    uint64_t redCheck[256];
    uint64_t greenCheck[256];
    uint64_t blueCheck[256];
    int errorFlag = 0;

    memset(&(redCheck[0]), 0, sizeof(uint64_t) * 256);
    memset(&(greenCheck[0]), 0, sizeof(uint64_t) * 256);
    memset(&(blueCheck[0]), 0, sizeof(uint64_t) * 256);
   
    for (int i=*data_pos; i < finfo.st_size; i+=3) 
    {      
      unsigned char *val = (unsigned char *)&(fdata[i]);
      blueCheck[*val]++;
      
      val = (unsigned char *)&(fdata[i+1]);
      greenCheck[*val]++;
      
      val = (unsigned char *)&(fdata[i+2]);
      redCheck[*val]++;   
    }

    for (int i = 0; i < 256; ++i)
    {
      if (redCheck[i] != redCount[i]) 
      {
        std::cout << "Index " << i << " | Wrong answer for red: " << redCount[i] << " (expected " << redCheck[i] << ")" << std::endl;
        errorFlag = 1;
      }
      if (greenCheck[i] != greenCount[i]) 
      {
        std::cout << "Index " << i << " | Wrong answer for green: " << greenCount[i] << " (expected " << greenCheck[i] << ")" << std::endl;
        errorFlag = 1;
      }
      if (blueCheck[i] != blueCount[i]) 
      {
        std::cout << "Index " << i << " | Wrong answer for blue: " << blueCount[i] << " (expected " << blueCheck[i] << ")" << std::endl;
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
