/**
 * @file brightness.cpp
 * @brief Template for a Host Application Source File.
 *
 */

#include <stdio.h>
#include <strings.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <ctype.h>
#include <bits/stdc++.h>
#include <omp.h>

#include "../../../utilBaselines.h"

#define MINCOLORVALUE 0 // Sets the max value that any color channel can be in a given pixel
#define MAXCOLORVALUE 255 // Sets the max value that any color channel can be in a given pixel 

// Params ---------------------------------------------------------------------
typedef struct Params
{
  std::string inputFile;
  bool shouldVerify;
  int brightnessCoefficient;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./brightness.out [options]"
          "\n"
          "\n    -i    24-bit .bmp input file (default=uses 'sample1.bmp' from '/cpp-histogram/histogram_datafiles' directory)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n    -b    brightness coefficient value (default=20)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.inputFile = "../../../cpp-histogram/histogram_datafiles/sample1.bmp";
  p.shouldVerify = false;
  p.brightnessCoefficient = 20;

  int opt;
  while ((opt = getopt(argc, argv, "h:i:v:b:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
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

uint8_t truncate(uint8_t pixelValue, int brightnessCoefficient)
{
  return static_cast<uint8_t> (std::min(MAXCOLORVALUE, std::max(MINCOLORVALUE, pixelValue + brightnessCoefficient)));
}

void brightness(std::vector<uint8_t> &resultData, uint64_t imgDataBytes, int brightnessCoefficient)
{ 
  #pragma omp parallel for
  for (uint64_t i = 0; i < imgDataBytes; ++i) {
    resultData[i] = truncate(resultData[i], brightnessCoefficient);
  }
}

int main(int argc, char *argv[]) 
{      
  struct Params params = getInputParams(argc, argv);
  std::string fn = params.inputFile;
  std::cout << "Running brightness on CPU for input file : '" << fn << "'" << std::endl;

  int fd;
  uint64_t imgDataBytes, tempImgDataOffset;
  struct stat finfo;
  char *fdata;
  unsigned short *dataPos;
  int imgDataOffsetPosition;

  // Start data parsing
  if (!fn.substr(fn.find_last_of(".") + 1).compare("bmp") == 0)
  {
    // TODO: reading in other types of input files
    std::cout << "Need work reading in other file types" << std::endl;
    return 1;
  } 
  else 
  {
    fd = open(params.inputFile.c_str(), O_RDWR);
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
    fdata = static_cast<char *>(mmap(0, finfo.st_size + 1, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (fdata == 0) 
    {
      perror("Failed to memory map the file");
      return 1;
    }

    imgDataOffsetPosition = 10; // Start of image data, ignoring unneeded header data and info
    // Defined according to the assumed input file structure given
  
    dataPos = (unsigned short *)(&(fdata[imgDataOffsetPosition]));
    tempImgDataOffset = static_cast<uint64_t>(*(dataPos));
    imgDataBytes = static_cast<uint64_t> (finfo.st_size) - tempImgDataOffset;
  }
  // End data parsing

  printf("This file has %ld bytes of image data with a brightness coefficient of %d\n", imgDataBytes, params.brightnessCoefficient);

  std::vector<uint8_t> imgData(fdata + *dataPos, fdata + finfo.st_size);
  std::vector<uint8_t> resultData = imgData;
  
  // Start timing
  auto start = std::chrono::high_resolution_clock::now();

  brightness(resultData, imgDataBytes, params.brightnessCoefficient);

  // End Timing
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsedTime = end - start;
  std::cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << std::endl;

  if (params.shouldVerify)
  {  
    int errorFlag = 0;
    for (uint64_t i = 0; i < imgDataBytes; ++i) 
    { 
      // baseline calculation
      imgData[i] = truncate(imgData[i], params.brightnessCoefficient); 

      // comparison between baseline and OpenMP implementation from brightness()
      if (imgData[i] != resultData[i])
      {
        std::cout << "Wrong answer at index " << i << " | Wrong OpenMP answer = " << resultData[i] << " (CPU expected = " << imgData[i] << ")" << std::endl;
        errorFlag = 1;
      }
    }
    if (!errorFlag)
    {
      std::cout << "Correct!" << std::endl;
    }
  }

  munmap(fdata, finfo.st_size);
  close(fd);
   
  return 0;
}
