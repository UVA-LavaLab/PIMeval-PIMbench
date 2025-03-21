/**
 * @file game-of-life.cpp
 * @brief Template for a Host Application Source File.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <stdint.h>
#include <iomanip>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <vector>

#include "utilBaselines.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t width;
  uint64_t height;
  char *configFile;
  char *inputFile;
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
          "\n");
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.width = 2048;
  p.height = 2048;
  p.configFile = nullptr;
  p.inputFile = nullptr;

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
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

/**
 * @brief cpu game of life kernel
 */
void game_of_life(const std::vector<std::vector<uint8_t>> &x, std::vector<std::vector<uint8_t>> &y) 
{
  int height = x.size();
  int width = x[0].size();
  uint8_t sum_cpu = x[0][1] + x[1][0] + x[1][1];
  uint8_t res_cpu = (uint8_t)(sum_cpu == 3);
  sum_cpu = (uint8_t)(sum_cpu == 2);
  sum_cpu &= x[0][0];
  res_cpu |= sum_cpu;
  y[0][0] = res_cpu;
  #pragma omp parallel for
  for(int i=1; i< (int) x[0].size()-1; ++i) {
    sum_cpu = x[0][i-1] + x[0][i+1] + x[1][i-1] + x[1][i] + x[1][i+1];
    res_cpu = (uint8_t)(sum_cpu == 3);
    sum_cpu = (uint8_t)(sum_cpu == 2);
    sum_cpu &= x[0][i];
    res_cpu |= sum_cpu;
    y[0][i] = res_cpu;
  }
  sum_cpu = x[0][width-2] + x[1][width-2] + x[1][width-1];
  res_cpu = (uint8_t)(sum_cpu == 3);
  sum_cpu = (uint8_t)(sum_cpu == 2);
  sum_cpu &= x[0][0];
  res_cpu |= sum_cpu;
  y[0][width-1] = res_cpu;

  #pragma omp parallel for
  for(int i=1; i < height-1; ++i) {
      sum_cpu = x[i-1][0] + x[i-1][1] + x[i][1] + x[i+1][0] + x[i+1][1];
      res_cpu = (uint8_t)(sum_cpu == 3);
      sum_cpu = (uint8_t)(sum_cpu == 2);
      sum_cpu &= x[i][0];
      res_cpu |= sum_cpu;
      y[i][0] = res_cpu;
      for(int j=1; j< width-1; ++j) {
        sum_cpu = x[i][j-1] + x[i][j+1] + x[i-1][j-1] + x[i-1][j] + x[i-1][j+1] + x[i+1][j-1] + x[i+1][j] + x[i+1][j+1];
        res_cpu = (uint8_t)(sum_cpu == 3);
        sum_cpu = (uint8_t)(sum_cpu == 2);
        sum_cpu &= x[i][j];
        res_cpu |= sum_cpu;
        y[i][j] = res_cpu;
      }
      sum_cpu = x[i-1][width-2] + x[i-1][width-1] + x[i][width-2] + x[i+1][width-2] + x[i+1][width-1];
      res_cpu = (uint8_t)(sum_cpu == 3);
      sum_cpu = (uint8_t)(sum_cpu == 2);
      sum_cpu &= x[i][width-1];
      res_cpu |= sum_cpu;
      y[i][width-1] = res_cpu;
    }

    sum_cpu = x[height-2][0] + x[height-2][1] + x[height-1][1];
    res_cpu = (uint8_t)(sum_cpu == 3);
    sum_cpu = (uint8_t)(sum_cpu == 2);
    sum_cpu &= x[height-1][0];
    res_cpu |= sum_cpu;
    y[height-1][0] = res_cpu;
    #pragma omp parallel for
    for(int i=1; i< width-1; ++i) {
      sum_cpu = x[height-1][i-1] + x[height-1][i+1] + x[height-2][i-1] + x[height-2][i] + x[height-2][i+1];
      res_cpu = (uint8_t)(sum_cpu == 3);
      sum_cpu = (uint8_t)(sum_cpu == 2);
      sum_cpu &= x[height-1][i];
      res_cpu |= sum_cpu;
      y[height-1][i] = res_cpu;
    }
    sum_cpu = x[height-2][width-2] + x[height-2][width-1] + x[height-1][width-2];
    res_cpu = (uint8_t)(sum_cpu == 3);
    sum_cpu = (uint8_t)(sum_cpu == 2);
    sum_cpu &= x[height-1][width-1];
    res_cpu |= sum_cpu;
    y[height-1][width-1] = res_cpu;

}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char **argv)
{
  struct Params params = input_params(argc, argv);

  std::vector<std::vector<uint8_t>> x, y;

  srand((unsigned)time(NULL));
  x.resize(params.height);
  for(size_t i=0; i<params.height; ++i) {
    x[i].resize(params.width);
    for(size_t j=0; j<params.width; ++j) {
      x[i][j] = rand() & 1;
    }
  }

  y.resize(x.size());
  for(size_t i=0; i<y.size(); ++i) {
    y[i].resize(x[0].size());
  }

  auto start = std::chrono::high_resolution_clock::now();
  
  for (int32_t i = 0; i < WARMUP; i++)
  {
    game_of_life(x, y);
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsedTime = (end - start)/WARMUP;
  std::cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << std::endl;

  return 0;
}