#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdlib> 
#include <getopt.h>
#include "../../util.h"

// Params ---------------------------------------------------------------------
typedef struct Params
{
  char *dramConfigFile;
  char *referenceEncodingInputFile;
  char *queryEncodingInputFile;
  size_t topK; 
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./hdc.out [options]"
          "\n"
          "\n    -h    help"
          "\n    -k    top-k [default=5]"
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
  p.topK = 5; 

  int opt;
  while ((opt = getopt(argc, argv, "h:c:k:r:q:v:")) >= 0)
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
    case 'k': 
      p.topK = std::stoi(optarg);
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

std::vector<size_t> getTopKIndices(std::vector<int> const& vec, size_t k)
{
    // Vector to store pairs of (value, original index).
    std::vector<std::pair<int, size_t>> heapWithOriginIndex;

    // Initialize the vector of pairs.
    for (size_t i = 0; i < vec.size(); ++i)
    {
        heapWithOriginIndex.push_back({vec[i], i});
    }

    // Define the comparator for a max-heap.
    auto const heapComp = [](auto const& a, auto const& b) {
        return a.first < b.first; // Max-heap
    };

    // Create the heap.
    std::make_heap(heapWithOriginIndex.begin(), heapWithOriginIndex.end(), heapComp);

    // Retrieve the top-k indices.
    std::vector<size_t> topKIndices;
    for (size_t i = 0; i < k && !heapWithOriginIndex.empty(); ++i)
    {
        // Extract the index of the top element.
        topKIndices.push_back(heapWithOriginIndex.front().second);

        // Remove the top element from the heap.
        std::pop_heap(heapWithOriginIndex.begin(), heapWithOriginIndex.end(), heapComp);
        heapWithOriginIndex.pop_back();
    }

    return topKIndices;
}

std::vector<std::vector<size_t>> searchDatabase(std::vector<std::vector<int>>& queryEnc,
                                 std::vector<std::vector<int>>& refEnc,
                                 size_t topK, 
                                 bool shouldVerify) {
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
    
    if (shouldVerify)
    {
      cout << "[INFO] Starting verification......\n";
      std::vector<std::vector<int>> C(row, std::vector<int>(columnB, 0));
      for (uint64_t i = 0; i < row; ++i)
      {
        for (uint64_t j = 0; j < columnB; ++j)
        {
          for (uint64_t k = 0; k < columnA; ++k)
          {
            C[i][j] += queryEnc[i][k] * refEnc[j][k];
          }
        }
      }
      bool shouldContinue = true;
      for (uint64_t i = 0; i < row && shouldContinue; ++i)
      {
        for (uint64_t j = 0; j < columnB; ++j)
        {
          if (C[i][j] != dist[i][j])
          {
            std::cout << "[Error]: Incorrect Result.\nHost: " << C[i][j] << "\t PIM: " << dist[i][j] << "\n";
            shouldContinue = false;
            break;
          }
        }
      }
      std::cout << "[INFO] CPU and PIM results match.\n"; 
    }

    // Find the index of the top-k elements 
    size_t query_rows = queryEnc.size();
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<size_t>> pred; 
    for (size_t i = 0; i < query_rows; ++i) {
        pred.push_back(getTopKIndices(dist[i], topK));
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> hostElapsedTime = end - start;
    cout << "[INFO] Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << endl;
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

    std::vector<std::vector<size_t>> pred = searchDatabase(queryEnc, refEnc, params.topK, params.shouldVerify);
    pimShowStats();

    return 0;
}

