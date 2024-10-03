// Test: C++ version of K-Nearest Neighbor
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

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

#include "../../util.h"
#include "libpimeval.h"

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
          "\nUsage:  ./knn.out [options]"
          "\n"
          "\n    -n    number of data points (default=1024 points)"
          "\n    -m    number of test points (default=20 points)"
          "\n    -d    dimension (default=2)"
          "\n    -k    neighbors (default=5)"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing training datapoints (default=generates datapoints with random numbers)"
          "\n    -t    input file containing testing datapoints (default=generates datapoints with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.numDataPoints = 1024;
  p.numTestPoints = 20;
  p.dimension = 2;
  p.k = 5;
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

// Verification ---------------------------------------------------------------------

float compute_distance(const vector<vector<int>>& ref,
                      const vector<vector<int>>& query,
                      int           dim,
                      int           ref_index,
                      int           query_index) {
  int sum = 0;
  // the last dim is the target dim, so don't include it
  for (int d=0; d<dim; ++d) {
      const float diff = ref[d][ref_index] - query[d][query_index];
      sum += abs(diff);
  }
  return sum;
}

vector<pair<int, int>> findKSmallestWithIndices(const vector<int>& data, int k) {
    priority_queue<pair<int, int>> maxHeap;
    for(int i = 0; i < (int) data.size(); ++i) {
        if ((int) maxHeap.size() < k) {
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

void countLabelsAndClassify(uint64_t numPoints, uint64_t numTests, vector<vector<int>> &dataPoints, 
          vector<vector<int>> &distMat, 
          int adjusted_dim, 
          int k,
          vector<int> &testPredictions
          ) {

  #pragma omp parallel for schedule(static)
  for(int i = 0; i < (int) numTests; i++) {
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
    for (const auto& entry : kSmallest) {
      int index = entry.second;
      int label = dataPoints[adjusted_dim][index];
      if (labelCount[label] > maxCount) {
        maxCount = labelCount[label];
        bestLabel = label;
      }
    }

    // Assign the most frequent label to the test point
    testPredictions[i] = bestLabel;
  }
}

bool knn_test(vector<vector<int>> ref,
          const vector<vector<int>> query,
          int           dim,
          int           k,
          const vector<int> pimResult) {

  int ref_nb = ref[0].size();
  int query_nb = query[0].size();
  // Allocate local array to store all the distances / indexes for a given query point 
  vector<vector<int>> distMat(query_nb, vector<int>(ref_nb));
  vector<int> index(ref_nb);

  // Process one query point at a time
  for (int i=0; i<query_nb; ++i) {

    // Compute all distances / indexes for this point
    for (int j=0; j<ref_nb; ++j) {
      distMat[i][j]  = compute_distance(ref, query, dim, j, i);
      index[j] = j;
    }
  }

  // perform classification for this query point
  vector<int> testResults(query_nb);
  countLabelsAndClassify(ref_nb, query_nb, ref, distMat, dim-1, k, testResults);

  for(int i = 0; i < query_nb; i++) {
    if(testResults[i] != pimResult[i]) {
      printf("Query point at index %i failed. Pim classified this point as %i but actual classification is %i\n", i, pimResult[i], testResults[i]);
      return false;
    }
  }

  return true;

}

// PIM ---------------------------------------------------------------------

void allocatePimObject(uint64_t numOfPoints, int dimension, std::vector<PimObjId> &pimObjectList, PimObjId refObj)
{
  int idx = 0;
  if (refObj == -1)
  {
    PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numOfPoints, PIM_INT32);
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
    PimObjId obj = pimAllocAssociated(refObj, PIM_INT32);
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
  for (int i = 0; i < (int) pimObjectList.size(); i++)
  {
    PimStatus status = pimCopyHostToDevice((void *)dataPoints[i].data(), pimObjectList[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
  }
}


void runKNN(uint64_t numOfPoints, uint64_t numOfTests, int dimension, int k, vector<vector<int>> &dataPoints, vector<vector<int>> &testPoints, vector<int> &testPredictions)
{
  vector<PimObjId> dataPointObjectList(dimension);
  // allocate data into the PIM space, like declaring variable
  allocatePimObject(numOfPoints, dimension, dataPointObjectList, -1);
  copyDataPoints(dataPoints, dataPointObjectList);

  vector<PimObjId> resultObjectList(dimension);
  allocatePimObject(numOfPoints, dimension, resultObjectList, dataPointObjectList[0]);


  vector<vector<int>> distMat(numOfTests, vector<int>(numOfPoints));
  
  for(int j = 0; j < (int) numOfTests; ++j){
    for (int i = 0; i < dimension; ++i){
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

  int adjusted_dim = dimension - 1; // Force the user to specficy the target index to the last dimensions in the data and test points

  auto start = std::chrono::high_resolution_clock::now();
  
  countLabelsAndClassify(numOfPoints, numOfTests, dataPoints, distMat, adjusted_dim, k, testPredictions);

  auto end = std::chrono::high_resolution_clock::now();
  hostElapsedTime += (end - start);

  for (int i = 0; i < (int) resultObjectList.size(); ++i) {
    pimFree(resultObjectList[i]);
  }

  for (int i = 0; i < (int) dataPointObjectList.size(); ++i) {
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

  std::cout << "Running KNN on PIM for datapoints: " << params.numDataPoints << "\n";

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


  pimShowStats();
  cout << "Host elapsed time: " << fixed << setprecision(3) << hostElapsedTime.count() << " ms." << endl;

  if (params.shouldVerify)
  {
    if (!knn_test(dataPoints, testPoints, dim, k, testPredictions)) {
      cerr << "KNN verification failed!\n";
    } else {
      cout << "KNN was succesfully verified against host result\n";
    }
  }

  return 0;
}
