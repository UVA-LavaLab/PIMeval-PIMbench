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

__global__ void game_of_life(uint8_t* x, uint8_t* y, int width, int curr_rows_todo, int offset, int height_in_chunk)
{
    int x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int y_index_res = blockIdx.y * blockDim.y + threadIdx.y;

    int y_index = y_index_res + offset;

    if(x_index < width && y_index_res < curr_rows_todo) {
        uint8_t sum_gpu = 0;
        sum_gpu += get_with_default(x_index-1, y_index-1, width, height_in_chunk, x);
        sum_gpu += get_with_default(x_index-1, y_index, width, height_in_chunk, x);
        sum_gpu += get_with_default(x_index-1, y_index+1, width, height_in_chunk, x);
        sum_gpu += get_with_default(x_index, y_index-1, width, height_in_chunk, x);
        sum_gpu += get_with_default(x_index, y_index+1, width, height_in_chunk, x);
        sum_gpu += get_with_default(x_index+1, y_index-1, width, height_in_chunk, x);
        sum_gpu += get_with_default(x_index+1, y_index, width, height_in_chunk, x);
        sum_gpu += get_with_default(x_index+1, y_index+1, width, height_in_chunk, x);

        uint8_t cell_state = x[y_index * width + x_index];
        uint8_t res_gpu = (sum_gpu == 3) || ((sum_gpu == 2) && cell_state);

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

  size_t free_mem, total_mem;
  cudaError_t cuda_status = cudaMemGetInfo(&free_mem, &total_mem);
  if (cuda_status != cudaSuccess) {
      std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(cuda_status) << std::endl;
      return -1;
  }

  int grid_sz = sizeof(uint8_t) * params.width * params.height;
  size_t usable_mem = static_cast<size_t>(free_mem * 0.7);
  size_t gpu_to_alloc_each = std::min(usable_mem / 2, (size_t) grid_sz + 2);
  size_t rows_per_iteration = gpu_to_alloc_each / params.width;
  if(rows_per_iteration <= 2) {
    std::cerr << "Not enough GPU memory for game of life" << std::endl;
    return 1;
  }
  size_t rows_per_iteration_usable = rows_per_iteration - 2;
  size_t num_iteration = 1 + params.height / rows_per_iteration_usable;
  size_t rows_final_iteration = params.height - ((num_iteration - 1) * rows_per_iteration_usable);//params.height - (num_iteration * rows_per_iteration);
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

  for(int i = 0; i < num_iteration; ++i) {
    int start_row = i * rows_per_iteration_usable;
    int curr_rows_todo = (i+1 == num_iteration) ? rows_final_iteration : rows_per_iteration_usable;
    if(curr_rows_todo == 0) {
        continue;
    }
    int end_row = start_row + curr_rows_todo;
    int curr_alloc_rows = curr_rows_todo;
    size_t curr_grid_offset = start_row * params.width;
    int offset = 0;

    if(start_row != 0) {
        curr_grid_offset -= params.width;
        curr_alloc_rows += 1;
        offset = 1;
    }

    if(end_row != params.height) {
        curr_alloc_rows += 1;
    } else {
        // Last chunk: add extra row and zero it out
        curr_alloc_rows += 1;
        cudaMemset(gpu_x + (curr_alloc_rows - 1) * params.width, 0, params.width * sizeof(uint8_t));
    }

    size_t max_host_data_sz = (params.height * params.width * sizeof(uint8_t)) - curr_grid_offset * sizeof(uint8_t);
    size_t curr_alloc_sz = std::min((curr_alloc_rows - (end_row == params.height ? 1 : 0)) * params.width * sizeof(uint8_t), max_host_data_sz);

    errorCode = cudaMemcpy(gpu_x, x.data() + curr_grid_offset, curr_alloc_sz, cudaMemcpyHostToDevice);
    if (errorCode != cudaSuccess) {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    dim3 dimBlock(32, 32);
    dim3 dimGrid((params.width + dimBlock.x - 1) / dimBlock.x, (curr_rows_todo + dimBlock.y - 1) / dimBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float timeElapsed = 0;

    cudaEventRecord(start, 0);

    game_of_life<<<dimGrid, dimBlock>>>(gpu_x, gpu_y, params.width, curr_rows_todo, offset, curr_alloc_rows);

    errorCode = cudaGetLastError();
    if (errorCode != cudaSuccess) {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeElapsed, start, stop);

    total_time += timeElapsed;

    size_t curr_output_sz = curr_rows_todo * params.width * sizeof(uint8_t);

    errorCode = cudaMemcpy(y.data() + start_row * params.width, gpu_y, curr_output_sz, cudaMemcpyDeviceToHost);
    if (errorCode != cudaSuccess) {
        cerr << "Cuda Error: " << cudaGetErrorString(errorCode) << "\n";
        exit(1);
    }
  }

  printf("Execution time of game of life = %f ms\n", total_time);
   
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
          exit(1);
        }
      }
    }
    if(is_correct) {
      std::cout << "Correct for game of life!" << std::endl;
    }
  }

  return 0;
}