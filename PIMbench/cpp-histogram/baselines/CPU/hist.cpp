/**
 * @file hist.cpp
 * @brief Template for a Host Application Source File.
 *
 */

#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#include <chrono>

#include "../../../utilBaselines.h"

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

typedef struct {
   unsigned char *data;
   long dataPos;
   long dataLength;
   uint64_t redCount[NUMBINS];
   uint64_t greenCount[NUMBINS];
   uint64_t blueCount[NUMBINS];
} thread_arg_t;

void *histogram(void *arg) 
{ 
  uint64_t *redCount;
  uint64_t *greenCount;
  uint64_t *blueCount;
  int i;
  thread_arg_t *threadArg = (thread_arg_t *)arg;
  unsigned char *val;

  redCount = threadArg->redCount;
  greenCount = threadArg->greenCount;
  blueCount = threadArg->blueCount;
   
  for (i= threadArg->dataPos; 
       i < threadArg->dataPos + threadArg->dataLength; 
       i+=NUMCHANNELS) 
  {              
    val = &(threadArg->data[i]);
    blueCount[*val]++;
      
    val = &(threadArg->data[i+1]);
    greenCount[*val]++;
      
    val = &(threadArg->data[i+2]);
    redCount[*val]++;   
  }
  return (void *)0;
}

int main(int argc, char *argv[]) 
{      
  struct Params params = getInputParams(argc, argv);
  std::string fn = params.inputFile;
  std::cout << "Input file : '" << fn << "'" << std::endl;

  int i, j;
  int fd;
  uint64_t imgDataBytes;
  struct stat finfo;
  char *fdata;
  unsigned short *dataPos;
  pthread_t *pid;
  pthread_attr_t attr;
  thread_arg_t *arg;
  int numProcs, numPerThread, excess;
  int imgDataOffsetPosition, numPixels;

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

    imgDataOffsetPosition = 10; // Start of image data, ignoring unneeded header data and info
    // Defined according to the assumed input file structure given

    dataPos = (unsigned short *)(&(fdata[imgDataOffsetPosition]));
    imgDataBytes = static_cast<uint64_t> (finfo.st_size) - static_cast<uint64_t>(*(dataPos));
    numPixels = imgDataBytes / NUMCHANNELS;
  }
  // End data parsing

  printf("This file has %ld bytes of image data, %ld pixels\n", imgDataBytes, imgDataBytes / NUMCHANNELS);

  uint64_t redCount[NUMBINS], greenCount[NUMBINS], blueCount[NUMBINS];

  memset(&(redCount[0]), 0, sizeof(uint64_t) * NUMBINS);
  memset(&(greenCount[0]), 0, sizeof(uint64_t) * NUMBINS);
  memset(&(blueCount[0]), 0, sizeof(uint64_t) * NUMBINS);
   
  pthread_attr_init(&attr);
  pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
   
  numProcs = sysconf(_SC_NPROCESSORS_ONLN);

  if (numProcs < 1)
  {
    perror("Error during SYSCONF call");
    exit(1);
  }

  numPerThread = numPixels / numProcs;
  excess = numPixels % numProcs;
   
  pid = (pthread_t *)malloc(sizeof(pthread_t) * numProcs);
  if (pid == NULL)
  {
    perror("Error during MALLOC call");
    exit(1);
  }

  arg = (thread_arg_t *)calloc(sizeof(thread_arg_t), numProcs);
  if (arg == NULL)
  {
    perror("Error during CALLOC call");
    exit(1);
  }
   
  // Start timing
  auto start = std::chrono::high_resolution_clock::now();

  long currentPos = (long)(*dataPos);
  for (i = 0; i < numProcs; i++) 
  {
    arg[i].data = (unsigned char *)fdata;
    arg[i].dataPos = currentPos;
    arg[i].dataLength = numPerThread;
    if (excess > 0) 
    {
      arg[i].dataLength++;
      excess--;
    }
      
    arg[i].dataLength *= NUMCHANNELS;
    currentPos += arg[i].dataLength;

    memset(arg[i].redCount, 0, sizeof(uint64_t) * NUMBINS);
    memset(arg[i].greenCount, 0, sizeof(uint64_t) * NUMBINS);
    memset(arg[i].blueCount, 0, sizeof(uint64_t) * NUMBINS);
      
    pthread_create(&(pid[i]), &attr, histogram, (void *)(&(arg[i])));   
  }
   
  for (i = 0; i < numProcs; i++) 
  {
    pthread_join(pid[i] , NULL);   
  }
   
  for (i = 0; i < numProcs; i++) 
  {
    for (j = 0; j < NUMBINS; j++) 
    {
      redCount[j] += arg[i].redCount[j];
      greenCount[j] += arg[i].greenCount[j];
      blueCount[j] += arg[i].blueCount[j];
    }
  }

  // End Timing
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsedTime = end - start;
  std::cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << std::endl;

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
        std::cout << "Index " << i << " | Wrong PIM answer for redCount = " << redCount[i] << " (CPU expected = " << redCheck[i] << ")" << std::endl;
        errorFlag = 1;
      }
      if (greenCheck[i] != greenCount[i]) 
      {
        std::cout << "Index " << i << " | Wrong PIM answer for greenCount = " << greenCount[i] << " (CPU expected = " << greenCheck[i] << ")" << std::endl;
        errorFlag = 1;
      }
      if (blueCheck[i] != blueCount[i]) 
      {
        std::cout << "Index " << i << " | Wrong PIM answer for blueCount = " << blueCount[i] << " (CPU expected = " << blueCheck[i] << ")" << std::endl;
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
   
  free(pid);
  free(arg);
  pthread_attr_destroy(&attr);
   
  return 0;
}
