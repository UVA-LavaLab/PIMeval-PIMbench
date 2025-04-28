/* File:     brightness.cu
 * Purpose:  Implement brightness on a gpu using Thrust
 *
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include "utilBaselines.h"

#define MINCOLORVALUE 0 // Sets the max value that any color channel can be in a given pixel
#define MAXCOLORVALUE 255 // Sets the max value that any color channel can be in a given pixel 

// Thrust error-checking macro
#define THRUST_CHECK() \
do { \
    cudaError_t err = cudaDeviceSynchronize(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Thrust error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

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

uint8_t truncate(int pixelValue, int brightnessCoefficient)
{
  return static_cast<uint8_t> (std::min(MAXCOLORVALUE, std::max(MINCOLORVALUE, pixelValue + brightnessCoefficient)));
}

struct calculateBrightness 
{
  int brightnessCoefficient;

  __host__ __device__
  calculateBrightness(int coeff) 
    : brightnessCoefficient(coeff) {}

  __host__ __device__
  uint8_t operator()(uint8_t pixelValue) 
  {
    int newValue = pixelValue + brightnessCoefficient;
    if (newValue < MINCOLORVALUE) 
    {
      return MINCOLORVALUE;
    }
    else if (newValue > MAXCOLORVALUE)
    {
      return MAXCOLORVALUE;
    }
    return static_cast<uint8_t> (newValue);
  }
};

int main(int argc, char *argv[]) 
{      
  struct Params params = getInputParams(argc, argv);
  std::string fn = params.inputFile;
  std::cout << "Running brightness on GPU for input file : '" << fn << "'" << std::endl;

  // Begin data parsing
  int fd;
  uint64_t imgDataBytes, tempImgDataOffset;
  struct stat finfo;
  char *fdata;
  unsigned short *dataPos;
  int brightnessCoefficient = params.brightnessCoefficient;
  int imgDataOffsetPosition;

  if (fn.substr(fn.find_last_of(".")) != ".bmp")
  {
    // TODO
    // assuming inputs will be 24-bit .bmp files
    std::cout << "Need work reading in other file types" << std::endl;
    return 1;
  } 
  else
  {
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
    tempImgDataOffset = static_cast<uint64_t>(*(dataPos));
    imgDataBytes = static_cast<uint64_t> (finfo.st_size) - tempImgDataOffset;
  }
  // End data parsing

  printf("This file has %llu bytes of image data with a brightness coefficient of %d\n", imgDataBytes, brightnessCoefficient);

  std::vector<uint8_t> imgData(fdata + *dataPos, fdata + finfo.st_size), resultData(imgDataBytes);

  thrust::device_vector<uint8_t> thrustImgData = imgData;
  THRUST_CHECK();

  auto [timeElapsed, avgPower, energy] = measureCUDAPowerAndElapsedTime([&]() {
    thrust::transform(thrustImgData.begin(), thrustImgData.end(), thrustImgData.begin(), calculateBrightness(brightnessCoefficient));
    THRUST_CHECK();
  });

  printf("\nExecution time of birghtness = %f ms\n", timeElapsed);
  printf("Average Power = %f mW\n", avgPower);
  printf("Energy Consumption = %f mJ\n", energy);

  thrust::copy(thrustImgData.begin(), thrustImgData.end(), resultData.begin());
  THRUST_CHECK();

  if (params.shouldVerify)
  {  
    int errorFlag = 0;
    for (uint64_t i = 0; i < imgDataBytes; ++i) 
    {     
      imgData[i] = truncate(imgData[i], params.brightnessCoefficient); 
      if (imgData[i] != resultData[i])
      {
        std::cout << "Wrong answer at index " << i << " | Wrong Thrust answer = " << resultData[i] << " (CPU expected = " << imgData[i] << ")" << std::endl;
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