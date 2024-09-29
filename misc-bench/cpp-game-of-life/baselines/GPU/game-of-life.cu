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
    return x[y_index * width + x_index];
  }
  return 0;
}

__global__ void game_of_life(uint8_t* x, uint8_t* y, int width, int height, int start_row, int end_row)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index_res = blockIdx.y * blockDim.y + threadIdx.y;
    int y_index = (start_row == 0) ? y_index_res : y_index_res + 1;
    int curr_height = end_row - start_row;
    if(x_index >= 0 && x_index <= width-1 && y_index >= 0 && y_index <= curr_height-1) {
      uint8_t sum_gpu = get_with_default(x_index-1, y_index-1, width, curr_height, x);
      sum_gpu += get_with_default(x_index-1, y_index, width, curr_height, x);
      sum_gpu += get_with_default(x_index-1, y_index+1, width, curr_height, x);
      sum_gpu += get_with_default(x_index, y_index-1, width, curr_height, x);
      sum_gpu += get_with_default(x_index, y_index+1, width, curr_height, x);
      sum_gpu += get_with_default(x_index+1, y_index-1, width, curr_height, x);
      sum_gpu += get_with_default(x_index+1, y_index, width, curr_height, x);
      sum_gpu += get_with_default(x_index+1, y_index+1, width, curr_height, x);
      uint8_t res_gpu = (uint8_t)(sum_gpu == 3);
      sum_gpu = (uint8_t)(sum_gpu == 2);
      sum_gpu &= x[y_index * width + x_index];
      res_gpu |= sum_gpu;
      y[y_index_res * width + x_index] = res_gpu;
    }
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Running GPU game of life for board: " << params.width << "x" << params.height << "\n";
  std::vector<uint8_t> x, y;
  if (params.inputFile == nullptr)
  {
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

  size_t free_memory = 0;
  size_t total_memory = 0;

  // Get free and total memory
  cudaError_t mem_result = cudaMemGetInfo(&free_memory, &total_memory);

  if (mem_result != cudaSuccess) {
      std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(mem_result) << std::endl;
      return 1;
  }

  std::cout << "Free memory: " << free_memory / (1024.0 * 1024.0) << " MB" << std::endl;
  std::cout << "Total memory: " << total_memory / (1024.0 * 1024.0) << " MB" << std::endl;

  int grid_sz = sizeof(uint8_t) * params.width * params.height;

  // Number below is a rough estimate from testing, TODO: look into better ways to get max memory alloc
  size_t gpu_max_sz = 100000000;
  size_t gpu_to_alloc_each = std::min(gpu_max_sz / 2, (size_t) grid_sz + 2);
  size_t rows_per_iteration = gpu_to_alloc_each / params.width;
  if(rows_per_iteration <= 2) {
    std::cerr << "Not enough GPU memory for game of life" << std::endl;
    return 1;
  }
  size_t rows_per_iteration_usable = rows_per_iteration - 2;
  size_t num_iteration = 1 + params.height / rows_per_iteration_usable;
  size_t rows_final_iteration = params.height - ((num_iteration - 1) * rows_per_iteration_usable);//params.height - (num_iteration * rows_per_iteration);

  cout << "gpu max sz: " << gpu_max_sz << " gpu to alloc each: " << gpu_to_alloc_each << " num iteration: " << num_iteration << " rows per iteration: " << rows_per_iteration_usable << " rows on final iteration: " << rows_final_iteration << endl;

  // cout << "to alloc: " << 2*gpu_to_alloc_each << ", gpu free: " << free_memory << endl;
  uint8_t* gpu_x;
  uint8_t* gpu_y;

  cudaError_t errorCode;

  errorCode = cudaMalloc((void**)&gpu_x, gpu_to_alloc_each);
  if (errorCode != cudaSuccess)
  {
      cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
      exit(1);
  }

  errorCode = cudaMalloc((void**)&gpu_y, gpu_to_alloc_each);
  if (errorCode != cudaSuccess)
  {
      cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
      exit(1);
  }

  double total_time = 0;
  cout << "num iter: " << num_iteration << endl;
  cout << "rows final iteration: " << rows_final_iteration << endl;

  for(int i=0; i<num_iteration; ++i) {
    // [start_row, end_row)
    int start_row = i*rows_per_iteration;
    int curr_rows_todo = (i+1 == num_iteration) ? rows_final_iteration : rows_per_iteration;
    if(curr_rows_todo == 0) {
      continue;
    }
    int end_row = start_row + curr_rows_todo;
    size_t curr_grid_offset = sizeof(uint8_t)*params.width*start_row;
    size_t curr_alloc_sz = sizeof(uint8_t)*params.width*curr_rows_todo;
    size_t curr_grid_offset_orig = curr_grid_offset;
    size_t curr_alloc_sz_orig = curr_alloc_sz;
    if(start_row != 0) {
      curr_grid_offset -= sizeof(uint8_t)*params.width;
      curr_alloc_sz += sizeof(uint8_t)*params.width;
    }
    if(end_row != params.height) {
      curr_alloc_sz += sizeof(uint8_t)*params.width;
    }
    errorCode = cudaMemcpy(gpu_x, x.data()+curr_grid_offset, curr_alloc_sz, cudaMemcpyHostToDevice);
    if (errorCode != cudaSuccess) {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    dim3 dimGrid (( params.width + 1023) / 1024 , curr_rows_todo , 1);
    dim3 dimBlock (1024 , 1 , 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float timeElapsed = 0;

    cudaEventRecord(start, 0);
    // cout << "206\n";

    game_of_life<<<dimGrid, dimBlock>>>(gpu_x, gpu_y, params.width, params.height, start_row, end_row);
    // cout << "209\n";
    
    errorCode = cudaGetLastError();
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeElapsed, start, stop);

    // printf("Execution time of game of life = %f ms\n", timeElapsed);
    total_time += timeElapsed;

    errorCode = cudaMemcpy(y.data() + curr_grid_offset_orig, gpu_y, curr_alloc_sz_orig, cudaMemcpyDeviceToHost);
    if (errorCode != cudaSuccess)
    {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }
    cout << "end of loop\n";
  }

  printf("Execution time of game of life = %f ms\n", total_time);
   
  cudaFree(gpu_x);
  cudaFree(gpu_y);
  cout << "after free\n";

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