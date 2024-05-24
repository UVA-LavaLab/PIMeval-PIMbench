// Test: C++ version of kmeans
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <unordered_map>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../util.h"
#include "libpimsim.h"

std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t numPoints;
  int maxItr;
  int dimension;
  int k;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./km [options]"
          "\n"
          "\n    -n    number of points (default=65536 elements)"
          "\n    -d    dimension (default=2)"
          "\n    -k    centroid (default=20)"
          "\n    -r    max iteration (default=2)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing datapoints (default=generates datapoints with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.numPoints = 65536;
  p.dimension = 2;
  p.k = 20;
  p.maxItr = 2;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:n:d:k:r:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'n':
      p.numPoints = strtoull(optarg, NULL, 0);
      break;
    case 'd':
      p.dimension = atoll(optarg);
      break;
    case 'k':
      p.k = atoll(optarg);
      break;
    case 'r':
      p.maxItr = atoll(optarg);
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

void allocatePimObject(uint64_t numOfPoints, int dimension, std::vector<PimObjId> &pimObjectList, PimObjId refObj)
{
  int idx = 0;
  unsigned bitsPerElement = sizeof(int) * 8;
  if (refObj == -1)
  {
    PimObjId obj1 = pimAlloc(PIM_ALLOC_V1, numOfPoints, bitsPerElement, PIM_INT32);
    if (obj1 == -1)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    pimObjectList[0] = obj1;
    refObj = obj1;
    idx = 1;
  }

  for (; idx < dimension; ++idx)
  {
    PimObjId obj = pimAllocAssociated(PIM_ALLOC_V1, numOfPoints, bitsPerElement, refObj, PIM_INT32);
    if (obj == -1)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
    pimObjectList[idx] = obj;
  }
}

void copyDataPoints(const std::vector<std::vector<int>> &dataPoints, std::vector<PimObjId> &pimObjectList)
{

  for (int i = 0; i < pimObjectList.size(); i++)
  {
    PimStatus status = pimCopyHostToDevice(PIM_COPY_V, (void *)dataPoints[i].data(), pimObjectList[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
  }
}

void copyCentroid(std::vector<int> &currCentroid, std::vector<PimObjId> &pimObjectList)
{
  for (int i = 0; i < pimObjectList.size(); i++)
  {
    PimStatus status = pimBroadCast(PIM_COPY_V, pimObjectList[i], currCentroid[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
  }
}

void runKmeans(uint64_t numOfPoints, int dimension, int k, int iteration, const std::vector<std::vector<int>> &dataPoints, std::vector<std::vector<int>> &centroids)
{
  std::vector<PimObjId> dataPointObjectList(dimension);
  allocatePimObject(numOfPoints, dimension, dataPointObjectList, -1);
  copyDataPoints(dataPoints, dataPointObjectList);

  std::vector<PimObjId> centroidObjectList(dimension);
  std::vector<PimObjId> resultObjectList(dimension);

  allocatePimObject(numOfPoints, dimension, centroidObjectList, dataPointObjectList[0]);
  allocatePimObject(numOfPoints, dimension, resultObjectList, dataPointObjectList[0]);
  // this object stores the minimum distance
  PimObjId tempObj = pimAllocAssociated(PIM_ALLOC_V1, numOfPoints, sizeof(int) * 8, resultObjectList[0], PIM_INT32);
  if (tempObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  for (int itr = 0; itr < iteration; ++itr)
  {
    std::vector<std::vector<int>> distMat(k, vector<int>(numOfPoints));
    std::vector<std::vector<int>> distFlag(k, vector<int>(numOfPoints));

    for (int i = 0; i < k; ++i)
    {
      copyCentroid(centroids[i], centroidObjectList);

      // for each centroid calculate manhattan distance. Not using euclidean distance to avoid multiplication.
      for (int idx = 0; idx < dimension; ++idx)
      {
        PimStatus status = pimSub(dataPointObjectList[idx], centroidObjectList[idx], resultObjectList[idx]);
        if (status != PIM_OK)
        {
          std::cout << "Abort" << std::endl;
          return;
        }
        status = pimAbs(resultObjectList[idx], resultObjectList[idx]);
        if (status != PIM_OK)
        {
          std::cout << "Abort" << std::endl;
          return;
        }
        if (idx > 0)
        {
          PimStatus status = pimAdd(resultObjectList[0], resultObjectList[idx], resultObjectList[0]);
          if (status != PIM_OK)
          {
            std::cout << "Abort" << std::endl;
            return;
          }
        }
      }
      PimStatus status = pimCopyDeviceToHost(PIM_COPY_V, resultObjectList[0], (void *)distMat[i].data());
      if (status != PIM_OK)
      {
        std::cout << "Abort" << std::endl;
      }
      if (i == 0)
      {
        // this can be replaced with device to device api
        PimStatus status = pimCopyHostToDevice(PIM_COPY_V, (void *)distMat[i].data(), tempObj);
        if (status != PIM_OK)
        {
          std::cout << "Abort" << std::endl;
        }
      }
      else
      {
        PimStatus status = pimMin(resultObjectList[0], tempObj, tempObj);
        if (status != PIM_OK)
        {
          std::cout << "Abort" << std::endl;
          return;
        }
      }
    }

    for (int i = 0; i < k; i++)
    {
      PimStatus status = pimCopyHostToDevice(PIM_COPY_V, (void *)distMat[i].data(), resultObjectList[0]);
      if (status != PIM_OK)
      {
        std::cout << "Abort" << std::endl;
      }
      status = pimEQ(resultObjectList[0], tempObj, resultObjectList[0]);
      if (status != PIM_OK)
      {
        std::cout << "Abort" << std::endl;
        return;
      }
      status = pimCopyDeviceToHost(PIM_COPY_V, resultObjectList[0], (void *)distFlag[i].data());
      if (status != PIM_OK)
      {
        std::cout << "Abort" << std::endl;
      }
    }

    // update the cluster in host
    // TODO: check if PIM will be benificial. My assumption it won't
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < k; i++)
    {
      std::fill(centroids[i].begin(), centroids[i].end(), 0);
    }
    std::vector<int> clusterPointCount(k, 0);
    for (int i = 0; i < k; ++i)
    {
      for (int j = 0; j < numOfPoints; ++j)
      {
        if (distFlag[i][j] == 1)
        {
          ++clusterPointCount[i];
          for (int d = 0; d < dimension; ++d)
          {
            centroids[i][d] += dataPoints[d][j];
          }
        }
      }
    }
    for (int i = 0; i < k; i++)
    {
      for (int j = 0; j < dimension; j++)
      {
        centroids[i][j] /= clusterPointCount[i];
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    hostElapsedTime += (end - start);
  }
  pimFree(tempObj);
  for (int i = 0; i < resultObjectList.size(); ++i)
  {
    pimFree(resultObjectList[i]);
  }

  for (int i = 0; i < dataPointObjectList.size(); ++i)
  {
    pimFree(dataPointObjectList[i]);
  }

  for (int i = 0; i < centroidObjectList.size(); ++i)
  {
    pimFree(centroidObjectList[i]);
  }
}

void initCentroids(int k, int dimension, int numOfPoints, std::vector<std::vector<int>> &centroids, const std::vector<std::vector<int>> &dataPoints)
{
  centroids.resize(k, vector<int>(dimension));

  for (int i = 0; i < k; i++)
  {
    int idx = rand() % numOfPoints;
    for (int j = 0; j < dimension; j++)
    {
      centroids[i][j] = dataPoints[j][idx];
    }
  }
}

// TODO: This implementation does not handle dimension that's large enough to not fit into one column. As in, bitsperelement*dimension*2 has to be smaller than the number of rows in a subarray.

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Number of points: " << params.numPoints << "\n";
  // row = dimension, col = number of datapoints. this is done to simplify data movement.
  std::vector<std::vector<int>> dataPoints;
  if (params.inputFile == nullptr)
  {
    getMatrix(params.dimension, params.numPoints, 0, dataPoints);
  }
  else
  {
    // TODO: Read from files
  }

  if (!createDevice(params.configFile))
    return 1;

  std::vector<std::vector<int>> centroids;
  initCentroids(params.k, params.dimension, params.numPoints, centroids, dataPoints);

  runKmeans(params.numPoints, params.dimension, params.k, params.maxItr, dataPoints, centroids);

  if (params.shouldVerify)
  {
  }

  pimShowStats();
  cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << endl;

  return 0;
}
