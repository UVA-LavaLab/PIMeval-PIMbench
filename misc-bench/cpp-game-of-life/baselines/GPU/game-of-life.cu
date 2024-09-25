/* File:     game-of-life.cu
 * Purpose:  Implement game of life on a gpu using cuda
 *
 */

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#include <cassert>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "utilBaselines.h"

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t width;
  uint64_t height;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./game-of-life.out [options]"
          "\n"
          "\n    -x    board width (default=2048 elements)"
          "\n    -y    board height (default=2048 elements)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing a game board (default=generates board with random states)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.width = 2048;
  p.height = 2048;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:x:y:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'x':
      p.width = strtoull(optarg, NULL, 0);
      break;
    case 'y':
      p.height = strtoull(optarg, NULL, 0);
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

__inline__ __host__ __device__ uint8_t get_with_default(int x_index, int y_index, size_t width, size_t height, uint8_t* x) {
  if(x_index >= 0 && x_index < width && y_index >= 0 && y_index < height) {
    return x[y_index * width + y_index];
  }
  return 0;
}

__host__ __device__ uint8_t get(int x_index, int y_index, size_t width, uint8_t* x) {
  return x[y_index * width + y_index];
}

__global__ void game_of_life(uint8_t* x, uint8_t* y, int width, int height)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index = blockIdx.y * blockDim.y + threadIdx.y;
    if(x_index >= 0 && x_index <= width-1 && y_index >= 0 && y_index <= height-1) {
      uint8_t sum_gpu = get_with_default(x_index-1, y_index-1, width, height, x);
      sum_gpu += get_with_default(x_index-1, y_index, width, height, x);
      sum_gpu += get_with_default(x_index-1, y_index+1, width, height, x);
      sum_gpu += get_with_default(x_index, y_index-1, width, height, x);
      sum_gpu += get_with_default(x_index, y_index+1, width, height, x);
      sum_gpu += get_with_default(x_index+1, y_index-1, width, height, x);
      sum_gpu += get_with_default(x_index+1, y_index, width, height, x);
      sum_gpu += get_with_default(x_index+1, y_index+1, width, height, x);
      uint8_t res_gpu = (uint8_t)(sum_gpu == 3);
      sum_gpu = (uint8_t)(sum_gpu == 2);
      sum_gpu &= x[y_index * width + x_index];
      res_gpu |= sum_gpu;
      y[y_index * width + x_index] = res_gpu;
    }
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  // params.width = 5;
  // params.height = 5;
  std::cout << "Running GPU game of life for board: " << params.width << "x" << params.height << "\n";
  std::vector<uint8_t> x, y;
  if (params.inputFile == nullptr)
  {
    // x = {1,0,1,0,0,
    //      0,1,1,1,0,
    //      0,0,0,1,1,
    //      1,1,0,1,0,
    //      1,0,0,1,1};

    // x = {{1,0,1,0,0},
    //      {0,1,1,1,0},
    //      {0,0,0,1,1},
    //      {1,1,0,1,0},
    //      {1,0,0,1,1}};
         //0, 1, 0, 0, 1, 
         // 1, 0, 0, 0, 1,

          // Correct Board
          // 0, 0, 1, 1, 0, 
          // 0, 1, 0, 0, 1, 
          // 1, 0, 0, 0, 1, 
          // 1, 1, 0, 0, 0, 
          // 1, 1, 1, 1, 1
    srand((unsigned)time(NULL));
    x.resize(params.height * params.width);
    for(size_t i=0; i<x.size(); ++i) {
      x[i] = rand() & 1;
    }
  } 
  else 
  {
    std::cout << "Reading from input file is not implemented yet." << std::endl;
    return 1;
  }
  
  y.resize(x.size());
  
  dim3 dimGrid (( params.width + 1023) / 1024 , params.height , 1);
  dim3 dimBlock (1024 , 1 , 1);

  uint8_t* gpu_x;
  uint8_t* gpu_y;

  int grid_sz = sizeof(uint8_t) * params.width * params.height;

  cudaError_t errorCode;

  errorCode = cudaMalloc((void**)&gpu_x, grid_sz);
  if (errorCode != cudaSuccess)
  {
      cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
      exit(1);
  }

  errorCode = cudaMalloc((void**)&gpu_y, grid_sz);
  if (errorCode != cudaSuccess)
  {
      cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
      exit(1);
  }
  
  errorCode = cudaMemcpy(gpu_x, x.data(), grid_sz, cudaMemcpyHostToDevice);
  if (errorCode != cudaSuccess) {
      cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
      exit(1);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float timeElapsed = 0;

  cudaEventRecord(start, 0);

  game_of_life<<<dimGrid, dimBlock>>>(gpu_x, gpu_y, params.width, params.height);
  
  errorCode = cudaGetLastError();
  if (errorCode != cudaSuccess)
  {
      cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
      exit(1);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&timeElapsed, start, stop);

  printf("Execution time of game of life = %f ms\n", timeElapsed);

  errorCode = cudaMemcpy(y.data(), gpu_y, grid_sz, cudaMemcpyDeviceToHost);
  if (errorCode != cudaSuccess)
  {
      cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
      exit(1);
  }
  
  cudaFree(gpu_x);
  cudaFree(gpu_y);

  if (params.shouldVerify) 
  {
    bool is_correct = true;
    for(int i=0; i<params.height; ++i) {
      for(int j=0; j<params.width; ++j) {
        uint8_t sum_cpu = get_with_default(j-1, i-1, params.width, params.height, x.data());
        sum_cpu += get_with_default(j-1, i, params.width, params.height, x.data());
        sum_cpu += get_with_default(j-1, i+1, params.width, params.height, x.data());
        sum_cpu += get_with_default(j, i-1, params.width, params.height, x.data());
        sum_cpu += get_with_default(j, i+1, params.width, params.height, x.data());
        sum_cpu += get_with_default(j+1, i-1, params.width, params.height, x.data());
        sum_cpu += get_with_default(j+1, i, params.width, params.height, x.data());
        sum_cpu += get_with_default(j+1, i+1, params.width, params.height, x.data());

        uint8_t res_cpu = (sum_cpu == 3) ? 1 : 0;
        sum_cpu = (sum_cpu == 2) ? 1 : 0;
        sum_cpu &= x[i * params.width + j];
        res_cpu |= sum_cpu;

        if (res_cpu != y[i * params.width + j])
        {
          std::cout << "Wrong answer: " << unsigned(y[i * params.width + j]) << " (expected " << unsigned(res_cpu) << ") at " << i << ", " << j << std::endl;
          is_correct = false;
        }
      }
    }
    if(is_correct) {
      std::cout << "Correct for game of life!" << std::endl;
    }
  }

  return 0;
}