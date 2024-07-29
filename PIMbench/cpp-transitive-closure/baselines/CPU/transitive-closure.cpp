/**
 * @file transitive-closure.cpp
 * @brief Template for a Host Application Source File.
 *
 */

#include <iostream>
#include <vector>
#include <getopt.h>
#include <iomanip>
#include <chrono>
#include <omp.h>

#include "../../../utilBaselines.h"
#include "../../transitive-closure.hpp"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t dataSize;
  int sparsityRate;
  char *configFile;
  std::string inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./transitive-closure.out [options]"
          "\n"
          "\n    -l    number of vertices for a generated dataset (default=2^8 vertices)"
          "\n    -r    sparsity rate n for a generated dataset where 0 < n < 100. (default=50 percent)"
          "\n    -c    dramsim config file"
          "\n    -i    specify a .csv input file containing the number of vertices followed by a 2D array, used for verification with CPU/GPU benchmarks"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.dataSize = 256; 
  p.sparsityRate = 50;
  p.configFile = nullptr;
  p.inputFile = "";
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:l:r:c:i:v:")) >= 0)
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
    case 'r':
      p.sparsityRate = strtoull(optarg, NULL, 0);
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

void transitiveClosure(std::vector<std::vector<int>> &adjMatrix, int numVertices) 
{
  for (int k = 0; k < numVertices; ++k) 
  {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < numVertices; ++i)
    {
      for (int j = 0; j < numVertices; ++j) 
      {
        if (adjMatrix[i][j] > adjMatrix[i][k] + adjMatrix[k][j]) 
        {
          adjMatrix[i][j] = adjMatrix[i][k] + adjMatrix[k][j];
        }
      }
    }
  }
}

int main(int argc, char *argv[]) 
{      
  struct Params params = getInputParams(argc, argv);
  
  std::vector<int> inputList;
  int numVertices;
  std::vector<std::vector<int>> generatedMatrix;

  if (params.inputFile.compare("") == 0)
  {
    numVertices = params.dataSize;
    getMatrix(numVertices, numVertices, generatedMatrix);
    srand(time(NULL));
    for (int i = 0; i < numVertices; ++i)
    {
      for (int j = 0; j < numVertices; ++j)
      {
        int sparsityChance = rand() % 100 + 1;
        if ((generatedMatrix[i][j] == 0 || sparsityChance <= params.sparsityRate) && (i != j))
        {
          generatedMatrix[i][j] = MAX_EDGE_VALUE; 
        }
        else if (i == j)
        {
          generatedMatrix[i][j] = 0;
        }
      }
    }

    std::cout << "Sparsity rate: " << params.sparsityRate << "%" << std::endl;
  }
  else 
  {
    if (!readCSV(params.inputFile, inputList, numVertices))
    {
      std::cout << "Failed to read input file or does not exist" << std::endl;
      return 1;
    }

    std::cout << "Input file: '" << params.inputFile << "'" << std::endl;

    generatedMatrix.resize(numVertices, std::vector<int>(numVertices));
    for (int i = 0; i < numVertices; ++i)
    {
      for (int j = 0; j < numVertices; ++j)
      {
        generatedMatrix[i][j] = inputList[(i * numVertices) + j];
      }
    }
  }

  std::cout << "Number of vertices: " << numVertices << std::endl;

  std::vector<std::vector<int>> cpuAdjMatrix (numVertices, std::vector<int>(numVertices));

  cpuAdjMatrix = generatedMatrix;

  // Start timing
  auto start = std::chrono::high_resolution_clock::now();

  transitiveClosure(generatedMatrix, numVertices);
  
  // End Timing
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsedTime = end - start;
  std::cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << std::endl;

  for (int i = 0; i < numVertices; ++i)
  {
    generatedMatrix[i][i] = 0;
  }      

  if (params.shouldVerify)
  {
    for (int k = 0; k < numVertices; ++k) 
    {
      for (int i = 0; i < numVertices; ++i) 
      {
        for (int j = 0; j < numVertices; ++j) 
        {
          if (cpuAdjMatrix[i][j] > (cpuAdjMatrix[i][k] + cpuAdjMatrix[k][j]))
          {  
            cpuAdjMatrix[i][j] = cpuAdjMatrix[i][k] + cpuAdjMatrix[k][j];
          }
        }
      }
    }

    int errorFlag = 0;
    for (int i = 0; i < numVertices; ++i)
    {
      for (int j = 0; j < numVertices; ++j)
      {
        if (cpuAdjMatrix[i][j] != generatedMatrix[i][j])
        {
          std::cout << "Incorrect result at index [" << i << "," << j << "] = " << generatedMatrix[i][j] << " (CPU expected = " << cpuAdjMatrix[i][j] << ")" << std::endl;
          errorFlag = 1;
        }
      }
    }

    if (!errorFlag) 
    {
      std::cout << "Correct!" << std::endl;
    }
  }

  return 0;
}
