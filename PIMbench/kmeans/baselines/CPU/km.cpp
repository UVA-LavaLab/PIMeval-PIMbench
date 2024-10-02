/**
 * @file km.cpp
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
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <chrono>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <cfloat>

#include "../../../utilBaselines.h"

using namespace std;

// Initializing global variables
vector<vector<int32_t>> dataPoints;
vector<vector<int32_t>> clusters;
unordered_map<int, vector<int>> clusterPointMap;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t numPoints;
  int maxItr;
  int dimension;
  int k;
  int numThreads;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./km.out [options]"
          "\n"
          "\n    -t    # of threads (default=8)"
          "\n    -p    number of points (default=1024 points)"
          "\n    -k    value of K (default=20)"
          "\n    -d    number of features (default=2 dimensions)"
          "\n    -i    max iteration (default=5 iteration)"
          "\n");
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.numPoints = 1024;
  p.k = 20;
  p.dimension = 2;
  p.numThreads = 8;
  p.maxItr = 5;

  int opt;
  while ((opt = getopt(argc, argv, "p:k:d:i:t:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'p':
      p.numPoints = atoll(optarg);
      break;
    case 'k':
      p.k = atoi(optarg);
      break;
    case 'd':
      p.dimension = atoi(optarg);
      break;
    case 't':
      p.numThreads = atoi(optarg);
      break;
    case 'i':
      p.maxItr = atoi(optarg);
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

inline int32_t calculateDistance(const vector<int32_t> &pointA, const vector<int32_t> &pointB, int dim) 
{
  int32_t sum = 0;
  for (int i = 0; i < dim; i++)
  {
    sum += abs(pointA[i] - pointB[i]);
  }
  return sum;
}

bool updateCluster(int k, int dim)
{
  bool hasChange = false;
  
  #pragma omp parallel for reduction(| : hasChange)
  for (int i = 0; i < k; ++i)
  {
    vector<double> newCluster(dim, 0.0);
    int clusterSize = clusterPointMap[i].size();
    if (clusterSize == 0)
    {
      continue;
    }        

    for (int idx : clusterPointMap[i])
    {
      for (int j = 0; j < dim; ++j)
      {
        newCluster[j] += dataPoints[idx][j];
      }
    }

    for (int j = 0; j < dim; ++j)
    {
      newCluster[j] /= clusterSize;
      if (clusters[i][j] != newCluster[j])
      {
        hasChange = true;
        clusters[i][j] = newCluster[j];
      }
    }
  }
  return hasChange;
}

void runKMeans(int numPoints, int k, int dim, int maxItr, int numThreads)
{
  omp_set_num_threads(numThreads);
  int itr = 0;
  bool hasChange = true;
  while (itr < maxItr && hasChange)
  {
    // Clear clusterPointMap efficiently
    for (auto &cluster : clusterPointMap)
    {
      cluster.second.clear();
    }

    #pragma omp parallel
    {
      // Use thread-private variables
      vector<vector<int>> localClusterMap(k);

      #pragma omp for schedule(static)
      for (int i = 0; i < numPoints; ++i)
      {
        int minK = 0;
        double minDis = DBL_MAX;

        for (int j = 0; j < k; ++j)
        {
          double dist = calculateDistance(dataPoints[i], clusters[j], dim);
          if (dist < minDis)
          {
            minDis = dist;
            minK = j;
          }
        }
        
        localClusterMap[minK].push_back(i);
      }

      #pragma omp critical
      {
        for (int j = 0; j < k; ++j)
        {
          clusterPointMap[j].insert(clusterPointMap[j].end(),
                                    localClusterMap[j].begin(),
                                    localClusterMap[j].end());
        }
      }
    }

    hasChange = updateCluster(k, dim);
    itr += 1;
  }
}

void initClusters(int k, int dimension, int numPoints)
{
  clusters.resize(k, vector<int32_t>(dimension));
  srand((unsigned)time(NULL));

  for (int i = 0; i < k; i++)
  {
    int idx = rand() % numPoints;
    for (int j = 0; j < dimension; j++)
    {
      clusters[i][j] = dataPoints[idx][j];
    }
  }
}

void printData(int32_t **dataArray, int row, int col)
{
  for (int i = 0; i < row; i++)
  {
    for (int j = 0; j < col; j++)
    {
      cout << dataArray[i][j] << "\t";
    }
    cout << "\n";
  }
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char **argv)
{
  struct Params p = input_params(argc, argv);

  int k = p.k, numPoints = p.numPoints, dim = p.dimension;

  getMatrix(numPoints, dim, dataPoints);
  initClusters(k, dim, numPoints);
  cout << "Set up done!\n";
  
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < WARMUP; i++)
  {
    runKMeans(numPoints, k, dim, p.maxItr, p.numThreads);
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsedTime = (end - start)/WARMUP;
  cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;

  return 0;
}
