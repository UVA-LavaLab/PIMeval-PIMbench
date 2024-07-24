// Test: C++ version of transitive closure
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <cassert>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../util.h"
#include "libpimeval.h"

using namespace std;

// max edge value acting as INF, can be set to a much greater value
#define MAX_EDGE_VALUE 10000  

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

bool readCSV(const std::string &filename, std::vector<int> &inputList, int &numVertices) 
{
  std::ifstream file(filename);
  if (!file.is_open()) 
  {
    std::cerr << "Error opening file: " << filename << std::endl;
    return false;
  }
    
  std::string line;
  if (std::getline(file, line)) {
    std::istringstream iss(line);
    iss >> numVertices;
  } 
  else 
  {
    std::cerr << "Error reading number of vertices from file: " << filename << std::endl;
    return false;
  }
    
  while (std::getline(file, line)) 
  {
    std::istringstream iss(line);
    std::string value;

    while (std::getline(iss, value, ',')) {
      if (value.find("inf") != std::string::npos) {
        inputList.push_back(MAX_EDGE_VALUE);
      } 
      else 
      {
        try 
        {
          inputList.push_back(std::stoi(value));
        } catch (const std::invalid_argument& e) {
          std::cerr << "Invalid value in file: " << value << std::endl;
          return false;
        } catch (const std::out_of_range& e) {
          std::cerr << "Value out of range in file: " << value << std::endl;
          return false;
        }
      }
    }
  }  

  file.close();
  return true;
}

void loadVector(std::vector<uint16_t> &keySegment, std::vector<uint16_t> &additionSegment, std::vector<uint16_t> adjList, int numVertices, int index)
{
  #pragma omp parallel for collapse(2) schedule(static)
  for (int i = 0; i < numVertices; ++i)
  {
    for (int j = 0; j < numVertices; ++j)
    {
      int segmentIndex = (i * numVertices) + j;

      keySegment[segmentIndex] = adjList[(i * numVertices) + index];
      additionSegment[segmentIndex] = adjList[(index * numVertices) + j];
    }
  }
}

void transitiveClosure(std::vector<uint16_t> &adjList, int numVertices)
{
  int bitsPerElement = sizeof(uint16_t) * 8;
  PimObjId adjListObj = pimAlloc(PIM_ALLOC_AUTO, numVertices * numVertices, bitsPerElement, PIM_UINT16);
  assert(adjListObj != -1);
  PimObjId keyObj = pimAllocAssociated(bitsPerElement, adjListObj, PIM_UINT16);
  assert(keyObj != -1);
  PimObjId additionObj = pimAllocAssociated(bitsPerElement, adjListObj, PIM_UINT16);
  assert(additionObj != -1);
  PimObjId tempObj = pimAllocAssociated(bitsPerElement, adjListObj, PIM_UINT16);
  assert(tempObj != -1);

  PimStatus status = pimCopyHostToDevice((void *) adjList.data(), adjListObj);
  assert(status == PIM_OK);

  std::vector<uint16_t> keySegment(numVertices * numVertices), additionSegment(numVertices * numVertices);
  
  for (int i = 0; i < numVertices; ++i)
  {
    // Timing of loadVector() not considered as functionality is possible using PIM architecture
    loadVector(keySegment, additionSegment, adjList, numVertices, i);

    status = pimCopyHostToDevice((void *) keySegment.data(), keyObj);
    assert(status == PIM_OK);

    status = pimCopyHostToDevice((void *) additionSegment.data(), additionObj);
    assert(status == PIM_OK);

    status = pimAdd(keyObj, additionObj, tempObj);  
    assert(status == PIM_OK);

    status = pimMin(tempObj, adjListObj, adjListObj);  
    assert(status == PIM_OK);

    status = pimCopyDeviceToHost(adjListObj, (void *) adjList.data());
    assert(status == PIM_OK);
  }

  pimFree(adjListObj);
  pimFree(keyObj);
  pimFree(additionObj);
  pimFree(tempObj);
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);

  std::vector<int> inputList;
  std::vector<std::vector<int>> generatedMatrix;
  int numVertices;
  std::vector<uint16_t> adjList; // Assuming no negative edge values in input and any self-directed and 
                                 //    non-existent edges = MAX_EDGE_VALUE (mocking as infinity)

  if (params.inputFile.compare("") == 0)
  {
    numVertices = params.dataSize;
    getMatrix(numVertices, numVertices, 0, generatedMatrix);
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

  for (int i = 0; i < numVertices; ++i)
  {
    for (int j = 0; j < numVertices; ++j)
    {
      adjList.push_back(static_cast<uint16_t> (generatedMatrix[j][i]));
    }
  }
  
  if (!createDevice(params.configFile))
  {
    return 1;
  }

  transitiveClosure(adjList, numVertices);

  std::vector<std::vector<uint16_t>> resultMatrix(numVertices, std::vector<uint16_t>(numVertices));
  for (int i = 0; i < numVertices; ++i)
  {
    for (int j = 0; j < numVertices; ++j)
    {
      resultMatrix[j][i] = adjList[(i * numVertices) + j];
    }
  }

  if (params.shouldVerify)
  {
    for (int k = 0; k < numVertices; k++) 
    {
      for (int i = 0; i < numVertices; i++) 
      {
        for (int j = 0; j < numVertices; j++) 
        {
          if ((generatedMatrix[i][j] > (generatedMatrix[i][k] + generatedMatrix[k][j])) 
               && (generatedMatrix[k][j] != MAX_EDGE_VALUE) 
               && (generatedMatrix[i][k] != MAX_EDGE_VALUE))
          {
            generatedMatrix[i][j] = generatedMatrix[i][k] + generatedMatrix[k][j];
          }
        }
      }
    }

    int errorFlag = 0;
    for (int i = 0; i < numVertices; ++i)
    {
      for (int j = 0; j < numVertices; ++j)
      {
        if (generatedMatrix[i][j] != resultMatrix[i][j])
        {
          std::cout << "Incorrect result at index [" << i << "," << j << "] = " << resultMatrix[i][j] << " (CPU expected = " << generatedMatrix[i][j] << ")" << std::endl;
          errorFlag = 1;
        }
      }
    }

    if (!errorFlag) 
    {
      std::cout << "Correct!" << std::endl;
    }
  }

  pimShowStats();

  return 0;
}
