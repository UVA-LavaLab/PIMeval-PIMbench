/* File:     hist.cu
 * Purpose:  Implement histogram on a gpu using CUB
 *
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <cub/cub.cuh>
 
using namespace std;

#define NUMBINS 256 // RGB values at any given pixel can be a value 0 to 255 (inclusive)
#define NUMCHANNELS 3 // Red, greenCount, and blueCount color channels

// Params ---------------------------------------------------------------------
typedef struct Params
{
  std::string inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./hist.out [options]"
          "\n"
          "\n    -i    24-bit .bmp input file (default=uses 'sample1.bmp' from the '/histogram_datafiles/' directory)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.inputFile = "../../histogram_datafiles/sample1.bmp";
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:i:v:")) >= 0)
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
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::string fn = params.inputFile;
  std::cout << "Running histogram on GPU for input file : '" << fn << "'" << std::endl;

  int fd;
  uint64_t imgDataBytes;
  struct stat finfo;
  char *fdata;
  unsigned short *dataPos;
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

  printf("This file has %llu bytes of image data, %llu pixels\n", imgDataBytes, imgDataBytes / NUMCHANNELS);

  std::vector<uint8_t> imgData(fdata + *dataPos, fdata + finfo.st_size);
  
  int cubDataPoints = 4; // Specified number of data points to perform a CUB histogram, 4th spot represents A in RGBA but is not needed in use case
  std::vector<uint8_t> h_samples((imgData.size() / NUMCHANNELS) * cubDataPoints);

  for (size_t i = 0, j = 0; i < imgData.size(); i+=NUMCHANNELS, j+=NUMCHANNELS + 1)
  {
    h_samples[j] = imgData[i];
    h_samples[j + 1] = imgData[i + 1];
    h_samples[j + 2] = imgData[i + 2];
    h_samples[j + 3] = 0; 
  }

  // End data parsing

  uint8_t* d_samples; 
  cudaError_t errorCode;

  errorCode = cudaMalloc(&d_samples, h_samples.size() * sizeof(uint8_t));
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }
  errorCode = cudaMemcpy(d_samples, h_samples.data(), h_samples.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  int num_samples = imgDataBytes / NUMCHANNELS;
  int *d_histogram[NUMCHANNELS];
  int num_levels[NUMCHANNELS] = {NUMBINS + 1, NUMBINS + 1, NUMBINS + 1};
  int lower_level[NUMCHANNELS] = {0, 0, 0};
  int upper_level[NUMCHANNELS] = {NUMBINS, NUMBINS, NUMBINS};

  for (int i = 0; i < NUMCHANNELS; ++i) 
  {
    errorCode = cudaMalloc(&d_histogram[i], num_levels[i] * sizeof(int));
    if (errorCode != cudaSuccess)
    {
      cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
      exit(1);
    }
    errorCode = cudaMemset(d_histogram[i], 0, num_levels[i] * sizeof(int));
    if (errorCode != cudaSuccess)
    {
      cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
      exit(1);
    }
  }

  int h_histogram[NUMCHANNELS][NUMBINS] = {0}; // h_histogram[2] is the red channel, h_histogram[1] is green, and h_histogram[0] is blue
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  std::cout << "Launching CUDA Kernel." << std::endl;

  // Event creation
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float timeElapsed = 0;

  // Start timer
  cudaEventRecord(start, 0);

  // Included as essential part in determing the temporary storage size for Histogram algorithm
  errorCode = cub::DeviceHistogram::MultiHistogramEven<NUMCHANNELS + 1, NUMCHANNELS>(d_temp_storage, temp_storage_bytes,
                        d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples);
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  errorCode = cudaMalloc(&d_temp_storage, temp_storage_bytes);
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  /* Kernel Call */
  errorCode = cub::DeviceHistogram::MultiHistogramEven<NUMCHANNELS + 1, NUMCHANNELS>(d_temp_storage, temp_storage_bytes,
  d_samples, d_histogram, num_levels, lower_level, upper_level, num_samples);
  if (errorCode != cudaSuccess)
  {
    cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
    exit(1);
  }

  // End timer
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeElapsed, start, stop);

  printf("Execution time = %f ms\n", timeElapsed);

  for (int i = 0; i < NUMCHANNELS; ++i) 
  {
    errorCode = cudaMemcpy (h_histogram[i], d_histogram[i], NUMBINS * sizeof(int), cudaMemcpyDeviceToHost);
    if (errorCode != cudaSuccess)
    {
      cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
      exit(1);
    }
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
      if (redCheck[i] != h_histogram[2][i]) 
      {
        std::cout << "Index " << i << " | Wrong CUB answer for red = " << h_histogram[2][i] << " (CPU expected = " << redCheck[i] << ")" << std::endl;
        errorFlag = 1;
      }
      if (greenCheck[i] != h_histogram[1][i]) 
      {
        std::cout << "Index " << i << " | Wrong CUB answer for green = " << h_histogram[1][i] << " (CPU expected = " << greenCheck[i] << ")" << std::endl;
        errorFlag = 1;
      }
      if (blueCheck[i] != h_histogram[0][i]) 
      {
        std::cout << "Index " << i << " | Wrong CUB answer for blue = " << h_histogram[0][i] << " (CPU expected = " << blueCheck[i] << ")" << std::endl;
        errorFlag = 1;
      }
    }
    if (!errorFlag)
    {
      std::cout << "Correct!" << std::endl;
    }
  }

  /* Free memory */
  cudaFree(d_samples); 
  cudaFree(d_temp_storage);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  for (int i = 0; i < NUMCHANNELS; ++i) 
  {
    cudaFree(d_histogram[i]);
  }

  return 0;
}
