// Test: C++ version of K-Nearest Neighbor
// Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <vector>
#include <queue>
#include <utility>
#include <algorithm>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#include <unordered_map>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../util.h"
#include "libpimsim.h"

using namespace std;

chrono::duration<double, milli> hostElapsedTime = chrono::duration<double, milli>::zero();

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t numTestPoints;
  uint64_t numDataPoints;
  int dimension;
  int k;
  char *configFile;
  char *inputTestFile;
  char *inputDataFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./knn [options]"
          "\n"
          "\n    -n    number of data points (default=65536 points)"
          "\n    -m    number of test points (default=100 points)"
          "\n    -d    dimension (default=2)"
          "\n    -k    neighbors (default=20)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing training datapoints (default=generates datapoints with random numbers)"
          "\n    -t    input file containing testing datapoints (default=generates datapoints with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.numDataPoints = 65536;
  p.numTestPoints = 100;
  p.dimension = 2;
  p.k = 20;
  p.configFile = nullptr;
  p.inputTestFile = nullptr;
  p.inputDataFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:n:m:d:k:c:i:t:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'n':
      p.numDataPoints = atoll(optarg);
      break;
    case 'm':
      p.numTestPoints = atoll(optarg);
      break;
    case 'd':
      p.dimension = atoll(optarg);
      break;
    case 'k':
      p.k = atoll(optarg);
      break;
    case 'c':
      p.configFile = optarg;
      break;
    case 'i':
      p.inputTestFile = optarg;
      break;
    case 't':
      p.inputDataFile = optarg;
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
    PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numOfPoints, bitsPerElement, PIM_INT32);
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
    PimObjId obj = pimAllocAssociated(bitsPerElement, refObj, PIM_INT32);
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
    PimStatus status = pimCopyHostToDevice((void *)dataPoints[i].data(), pimObjectList[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
  }
}

vector<pair<int, int>> findKSmallestWithIndices(const vector<int>& data, int k) {
    priority_queue<pair<int, int>> maxHeap;
    for(int i = 0; i < data.size(); ++i) {
        if (maxHeap.size() < k) {
            maxHeap.push({data[i], i});
        } else if (data[i] < maxHeap.top().first) {
            maxHeap.pop();
            maxHeap.push({data[i], i});
        }
    }

    vector<pair<int, int>> result(k);
    int idx = k-1;
    
    while(!maxHeap.empty() && idx >= 0) {
        result[idx--] = maxHeap.top();
        maxHeap.pop();
    }

    return result;   
}

void runKNN(uint64_t numOfPoints, uint64_t numOfTests, int dimension, int k, const vector<vector<int>> &dataPoints, vector<vector<int>> &testPoints, vector<int> &testPredictions)
{
  int adjusted_dim = dimension - 1; // Force the user to specficy the target index to the last column in the data and test points
  vector<PimObjId> dataPointObjectList(adjusted_dim);
  // allocate data into the PIM space, like declaring variable
  allocatePimObject(numOfPoints, adjusted_dim, dataPointObjectList, -1);
  copyDataPoints(dataPoints, dataPointObjectList);

  vector<PimObjId> resultObjectList(adjusted_dim);
  allocatePimObject(numOfPoints, adjusted_dim, resultObjectList, dataPointObjectList[0]);


  vector<vector<int>> distMat(numOfTests, vector<int>(numOfPoints));
  
  for(int j = 0; j < numOfTests; ++j){
    for (int i = 0; i < adjusted_dim; ++i){
      // for each point calculate manhattan distance. Not using euclidean distance to avoid multiplication.

      PimStatus status = pimSubScalar(dataPointObjectList[i], resultObjectList[i], testPoints[i][j]);
      if (status != PIM_OK)
      {
        cout << "Abort" << endl;
        return;
      }
      status = pimAbs(resultObjectList[i], resultObjectList[i]);
      if (status != PIM_OK)
      {
        cout << "Abort" << endl;
        return;
      }
      if (i > 0)
      {
        status = pimAdd(resultObjectList[0], resultObjectList[i], resultObjectList[0]);
        if (status != PIM_OK)
        {
          std::cout << "Abort" << std::endl;
          return;
        }
      }

    }

    PimStatus status = pimCopyDeviceToHost(resultObjectList[0], (void *)distMat[j].data());
    if (status != PIM_OK)
    {
      cout << "Abort" << endl;
    }
  }

  auto start = std::chrono::high_resolution_clock::now();
  int numTests = distMat.size();
#pragma omp parallel for schedule(static)
  for(int i = 0; i < numTests; i++) {
    vector<pair<int, int>> kSmallest = findKSmallestWithIndices(distMat[i], k);

    // Tally the labels of the k nearest neighbors
    unordered_map<int, int> labelCount;
    for (const auto& elem : kSmallest) {
      int index = elem.second;
      int label = dataPoints[adjusted_dim][index];
      #pragma omp atomic
      labelCount[label]++;
    }
    
    // Find the label with the highest count
    int maxCount = 0;
    int bestLabel = -1;
    for (const auto& entry : labelCount) {
      if (entry.second > maxCount) {
        maxCount = entry.second;
        bestLabel = entry.first;
      }
    }

    // Assign the most frequent label to the test point
    testPredictions[i] = bestLabel;
  }
  auto end = std::chrono::high_resolution_clock::now();
  hostElapsedTime += (end - start);

  for (int i = 0; i < resultObjectList.size(); ++i) {
    pimFree(resultObjectList[i]);
  }

  for (int i = 0; i < dataPointObjectList.size(); ++i) {
    pimFree(dataPointObjectList[i]);
  }

}

vector<vector<int>> readCSV(const string& filename) {
  vector<vector<int>> data;
  ifstream file(filename);
  string line;
  
  if (!file.is_open()) {
    throw runtime_error("Could not open file");
  }


  while (getline(file, line)) {
    vector<int> row;
    stringstream ss(line);
    string value;
    
    while (getline(ss, value, ',')) {
      try {
        int intValue = stoi(value);
        row.push_back(intValue);
      } catch (const invalid_argument& e) {
        cerr << "Invalid argument: " << e.what() << " for value " << value << '\n';
      } catch (const out_of_range& e) {
        cerr << "Out of range: " << e.what() << " for value " << value << '\n';
      }
    }
    data.push_back(row);
  }

  file.close();

  // Transpose the matrix
  if (data.empty()) {
      return data; // Return empty if no data
  }

  size_t numRows = data.size();
  size_t numCols = data[0].size();

  vector<vector<int>> transposedData(numCols, vector<int>(numRows));

  for (size_t i = 0; i < numRows; ++i) {
      for (size_t j = 0; j < numCols; ++j) {
          transposedData[j][i] = data[i][j];
      }
  }

  return transposedData;
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  vector<vector<int>> dataPoints;
  vector<vector<int>> testPoints;



  if (params.inputTestFile == nullptr)
  {
    getMatrix(params.dimension, params.numTestPoints, 0, testPoints);
  }
  else
  {
    vector<vector<int>> test_data_int = readCSV(params.inputTestFile);

    vector<vector<int>> test_data = test_data_int;
    params.dimension = test_data.size();
    params.numTestPoints = test_data[0].size();
    testPoints = test_data;
  }
  if (params.inputDataFile == nullptr)
  {
    getMatrix(params.dimension, params.numDataPoints, 0, dataPoints);
  }
  else
  {
    vector<vector<int>> train_data_int = readCSV(params.inputDataFile);

    vector<vector<int>> train_data = train_data_int;
    params.dimension = train_data.size();
    params.numDataPoints = train_data[0].size();

    dataPoints = train_data;
  }

  int k = params.k, numPoints = dataPoints[0].size(), numTests = testPoints[0].size(), dim = params.dimension;

  if (!createDevice(params.configFile))
    return 1;

  // load predictions into var
  vector<int> testPredictions(numTests);
  runKNN(numPoints, numTests, dim, params.k, dataPoints, testPoints, testPredictions);

  if (params.shouldVerify)
  {
  }

  pimShowStats();
  cout << "Host elapsed time: " << fixed << setprecision(3) << hostElapsedTime.count() << " ms." << endl;

  return 0;
}
