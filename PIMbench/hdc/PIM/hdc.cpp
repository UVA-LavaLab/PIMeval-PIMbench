#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <getopt.h>
#include "../../util.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  char *dramConfigFile;
  char *referenceEncodingInputFile;
  char *queryEncodingInputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./hdc.out [options]"
          "\n"
          "\n    -h    help"
          "\n    -v    should verify result with CPU"
          "\n    -r    input reference file"
          "\n    -q    input query file"
          "\n    -c    input file containing dramsim config"
          "\n");
}


struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.dramConfigFile = nullptr;
  p.referenceEncodingInputFile = nullptr;
  p.queryEncodingInputFile = nullptr;

  int opt;
  while ((opt = getopt(argc, argv, "h:c:r:q:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'c':
      p.dramConfigFile = optarg;
      break;
    case 'r':
      p.referenceEncodingInputFile = optarg;
      break;
    case 'q':
      p.queryEncodingInputFile = optarg;
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


std::vector<std::vector<int>> readCsvToVector(const std::string& filename) 
{
    std::ifstream file(filename);
    std::vector<std::vector<int>> data;
    
    if (!file.is_open()) 
    {
        throw std::runtime_error("Could not open file");
    }

    std::string line;
    if (std::getline(file, line))
    {
        std::istringstream dim_stream(line);
        int dim1, dim2;
        char comma;
        if (!(dim_stream >> dim1 >> comma >> dim2))
        {
            throw std::runtime_error("Invalid format for dimensions");
        }
    }

    while (std::getline(file, line))
    {
        std::istringstream data_stream(line);
        std::vector<int> row;
        float value;
        while (data_stream >> value)
        {
            row.push_back((int)value);
            if (data_stream.peek() == ',')
            {
                data_stream.ignore();
            }
        }
        data.push_back(row);
    }

    return data;
}


void gemv(uint64_t row, uint64_t col, std::vector<int> &srcVector, std::vector<std::vector<int>> &srcMatrix, std::vector<int> &dst)
{
  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, row, PIM_INT32);
  if (srcObj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  PimObjId srcObj2 = pimAllocAssociated(srcObj1, PIM_INT32);
  if (srcObj2 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId dstObj = pimAllocAssociated(srcObj1, PIM_INT32);
  if (dstObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimStatus status = pimBroadcastInt(dstObj, 0);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  for (uint64_t i = 0; i < col; ++i)
  {
    status = pimCopyHostToDevice((void *)srcMatrix[i].data(), srcObj1);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimMulScalar(srcObj1, srcObj2, srcVector[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimAdd(srcObj2, dstObj, dstObj);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
  }

  dst.reserve(row);
  status = pimCopyDeviceToHost(dstObj, (void *)dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
  }
  pimFree(srcObj1);
  pimFree(srcObj2);
  pimFree(dstObj);
}

void transposeMatrix(uint64_t row, uint64_t col, std::vector<std::vector<int>> &srcMatrix, std::vector<std::vector<int>> &dstMatrix)
{
#pragma omp parallel for
  for (uint64_t i = 0; i < col; ++i)
  {
    for (uint64_t j = 0; j < row; ++j)
    {
      dstMatrix[i][j] = srcMatrix[j][i];
    }
  }
}

void gemm(uint64_t row, uint64_t colA, uint64_t colB, std::vector<std::vector<int>> &srcMatrixA, std::vector<std::vector<int>> &srcMatrixB, std::vector<std::vector<int>> &dstMatrix)
{
  dstMatrix.resize(row, std::vector<int>(colB, 0));
  std::vector<std::vector<int>> transposedDstMat(colB, std::vector<int>(row, 0));
  vector<std::vector<int>> srcMatrixAT(colA, std::vector<int>(row, 0)), srcMatrixBT(colB, std::vector<int>(colA, 0));
  transposeMatrix(row, colA, srcMatrixA, srcMatrixAT);
  transposeMatrix(colA, colB, srcMatrixB, srcMatrixBT);
  for (uint64_t i = 0; i < colB; ++i)
  {
    gemv(row, colA, srcMatrixBT[i], srcMatrixAT, transposedDstMat[i]);
  }
  transposeMatrix(colB, row, transposedDstMat, dstMatrix);
}

std::vector<int> searchDatabase(std::vector<std::vector<int>>& queryEnc,
                                 std::vector<std::vector<int>>& refEnc) {
    std::cout << "[INFO] Searching database" << std::endl;
  
    std::vector<std::vector<int>> refEncTransposed;
    getMatrix(refEnc[0].size(), refEnc.size(), 0, refEncTransposed);
    transposeMatrix(refEnc.size(), refEnc[0].size(), refEnc, refEncTransposed);

    // Compute distances using the refactored function
    std::vector<std::vector<int>> dist; 
    uint64_t row = queryEnc.size(); 
    uint64_t columnA = queryEnc[0].size(); 
    uint64_t columnB = refEncTransposed[0].size(); 
    gemm(row, columnA, columnB, queryEnc, refEncTransposed, dist);

    // Find the index of the maximum value for each query
    size_t query_rows = queryEnc.size();
    std::vector<int> pred(query_rows, 0);

    for (size_t i = 0; i < query_rows; ++i) {
        pred[i] = std::max_element(dist[i].begin(), dist[i].end()) - dist[i].begin();
    }

    return pred;
}

int main(int argc, char *argv[]) 
{
    std::cout << "[INFO] Parsing input arguements" << std::endl;
    struct Params params = getInputParams(argc, argv);
    std::vector<std::vector<int>> refEnc = readCsvToVector(std::string(params.referenceEncodingInputFile));
    std::vector<std::vector<int>> queryEnc = readCsvToVector(std::string(params.queryEncodingInputFile));
    
    if (!createDevice(params.dramConfigFile))
      return 1;

    std::vector<int> pred = searchDatabase(queryEnc, refEnc);
    pimShowStats();

    return 0;
}

