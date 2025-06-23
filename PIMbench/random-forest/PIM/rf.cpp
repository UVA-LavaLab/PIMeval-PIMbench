// Test: C++ version of Decision Tree
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <atomic>
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

#include "util.h"
#include "libpimeval.h"

using namespace std;

chrono::duration<double, milli> hostElapsedTime = chrono::duration<double, milli>::zero();

const int LEQ = 0;
const int LT = 1;
const int GEQ = 2;
const int GT = 3;

// Params ---------------------------------------------------------------------

vector<vector<int>> negativeParameterMapping;
vector<vector<int>> modelParameter;
vector<vector<int>> compareParameterMapping;  // probably not needed once optimizations are implemented ()

typedef struct Params
{
    int tree_count;
    uint64_t dimension;
    int numThreads;
    int treeDepth;
    char *configFile;
} Params;

void usage()
{
    fprintf(stderr,
            "\nUsage:  ./knn.out [options]"
            "\n"
            "\n    -t    # of threads (default=8)"
            "\n    -n    number of trees (default=100 trees)"
            "\n    -d    input dimension (default=10)"
            "\n    -m    max tree depth (default=5)"
            "\n    -c    dramsim config file"
            "\n");
}

struct Params input_params(int argc, char **argv)
{
    struct Params p;
    p.dimension = 10;
    p.tree_count = 100;
    p.numThreads = 8;
    p.treeDepth = 5;
    p.configFile = nullptr;


    int opt;
    while ((opt = getopt(argc, argv, "h:t:n:m:d:c:")) >= 0)
    {
        switch (opt)
        {
        case 'h':
            usage();
            exit(0);
            break;
        case 'd':
            p.dimension = atoll(optarg);
            break;
        case 't':
            p.numThreads = atoi(optarg);
            break;
        case 'n':
            p.tree_count = atoi(optarg);
            break;
        case 'm':
            p.treeDepth = atoi(optarg);
            break;
        case 'c':
            p.configFile = optarg;
            break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
        }
    }

    return p;
}

void getModelAndInput(uint64_t numRows, uint64_t numCols, vector<vector<int>> &matrix, vector<vector<int>> &mappingMatrix, vector<vector<int>> &compareMatrix)
{
    // Seed the random number generator with a fixed seed for reproducibility

    // Resize the matrix to the specified number of rows and columns
    matrix.resize(numRows, vector<int>(numCols));
    mappingMatrix.resize(numRows, vector<int>(numCols));
    compareMatrix.resize(numRows, vector<int>(numCols));
    // eqMappingMatrix.resize(numRows, vector<int>(numCols));

    #pragma omp parallel for
    for (uint64_t row = 0; row < numRows; ++row) {
        for (uint64_t col = 0; col < numCols; ++col) {

            matrix[row][col] = ((row*col + row) % 2001) - 1000;  // Generates a number in [-1000, 1000]
            
            // to simulate compare operations at each leaf 
            int rand_op = matrix[row][col] % 4;  // reuse random value to generate a random sign
            compareMatrix[row][col] = rand_op;

            // if its > or >=, switch operators and negate parameter
            if (rand_op == GEQ or rand_op == GT) {
                mappingMatrix[row][col] = -1;  // indicates that input needs to be negated
                matrix[row][col] = -1 * matrix[row][col];  // negate mdoel parameter
            } else mappingMatrix[row][col] = 1;


        }
    }
}

// Verification ---------------------------------------------------------------------


int countLabelsAndClassify(uint64_t numPaths, uint64_t dim, uint64_t numTrees, vector<int> &compareResult, vector<int> &finalResult) {

  int numPathsDt = (numPaths / numTrees);  // number of paths PER decison tree in the ensemble/RF 
  std::vector<std::atomic<int>> labelCount(numPathsDt);

  
  // loop each decision tree in parallel
  //#pragma omp parallel for schedule(static)

  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for
  for(int i = 0; i < (int) numPaths; i += numPathsDt) {
    
    // Tally the labels of the k nearest neighbors
    int indRes = 0;
    int lastMatch = -1;
    for(int j = 0; j < numPathsDt; j++) {
      if (compareResult[i + j] == 1) {
        lastMatch = j; // since our implementation is based on random values, need to iterate through all elements
      }
    }

    if (lastMatch != -1) {
            labelCount[lastMatch]++;
        }

    
  }

  int maxValue = -1;
  int maxKey = -1;

  for (int i = 0; i < numPathsDt; ++i) {
        if (labelCount[i] > maxValue) {
            maxValue = labelCount[i];
            maxKey = i;
        }
  }
  auto end = std::chrono::high_resolution_clock::now();
  hostElapsedTime += (end - start);
  return maxKey;
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


int runRF(uint64_t numOfPaths, int numTrees, int dimension, vector<vector<int>> &inputFeatures, vector<vector<int>> &eqCompareOps, vector<int> &hostResult)
{
  // allocate and copy model parameters into PIM device 
  vector<PimObjId> modelParamObjectList(dimension);
  allocatePimObject(numOfPaths, dimension, modelParamObjectList, -1);
  copyDataPoints(modelParameter, modelParamObjectList);

  // allocate and copy input parameters into PIM device 
  vector<PimObjId> inputObjectList(dimension);
  allocatePimObject(numOfPaths, dimension, inputObjectList, modelParamObjectList[0]);
  copyDataPoints(modelParameter, inputObjectList);

  // allocate and copy equal operator mapping parameters into PIM device 
  vector<PimObjId> eqMappingObjectList(dimension);
  allocatePimObject(numOfPaths, dimension, eqMappingObjectList, modelParamObjectList[0]);
  copyDataPoints(eqCompareOps, eqMappingObjectList);

  PimObjId temp = pimAllocAssociated(modelParamObjectList[0], PIM_BOOL);  // for intermediate value store during logic statement
  PimObjId pimResult = pimAllocAssociated(temp, PIM_BOOL);  // initalize object to all zerosPimObjId temp = pimAllocAssociated(modelParamObjectList[0], PIM_BOOL);  // for intermediate value store during logic statement
  PimObjId temp1 = pimAllocAssociated(temp, PIM_BOOL);  // for intermediate value store during logic statement
  
  
  
  for(int i = 0; i < (int) dimension; ++i){
    // for each row/input dimension do the following evaluate the following logic statement:
    //      ( inputObjectList[i] < modelParamObjectList[i] ) or (eqMappingObjectList[i] == modelParamObjectList[i])
    PimStatus status = pimLT(inputObjectList[i], modelParamObjectList[i], temp);  // this will override preivous temp values
    if (status != PIM_OK)
    {
      cout << "Abort" << endl;
      return -1;
    }

    status = pimEQ(eqMappingObjectList[i], modelParamObjectList[i], temp1);
    if (status != PIM_OK)
    {
      cout << "Abort" << endl;
      return -1;
    }

    status = pimOr(temp1, temp, temp);
    if (status != PIM_OK)
    {
      cout << "Abort" << endl;
      return -1;
    }

    if (i == 0)
    {
      // if this is the first row, copy the result to the pimResult
      status = pimCopyObjectToObject(temp, pimResult);
      if (status != PIM_OK)
      {
        cout << "Abort" << endl;
        return -1;
      }
      continue;
    }
    status = pimAnd(temp, pimResult, pimResult);
    if (status != PIM_OK)
    {
      cout << "Abort" << endl;
      return -1;
    }
  }

  vector<int> pimCompareResult(numOfPaths);
  vector<uint8_t> pimCompareResultBool(numOfPaths, 0);  // initialize to false

  PimStatus status = pimCopyDeviceToHost(pimResult, (void *)pimCompareResultBool.data());
  if (status != PIM_OK)
  {
    cout << "Abort" << endl;
  }

  for (int i = 0; i < (int) pimCompareResult.size(); ++i) {
    // convert the boolean result to int
    if (pimCompareResultBool[i] == 1) {
      pimCompareResult[i] = 1;  // true
    } else {
      pimCompareResult[i] = 0;  // false
    }
  }
  
  int classificationResult = countLabelsAndClassify(numOfPaths, dimension, numTrees, pimCompareResult, hostResult);


  for (int i = 0; i < (int) modelParamObjectList.size(); ++i) {
    pimFree(modelParamObjectList[i]);
  }

  for (int i = 0; i < (int) inputObjectList.size(); ++i) {
    pimFree(inputObjectList[i]);
  }

  pimFree(pimResult);
  pimFree(temp);
  pimFree(temp1);

  return classificationResult;
}


int main(int argc, char *argv[])
{
  struct Params params = input_params(argc, argv);
  uint64_t numberOfPaths = ((uint64_t) pow(2, params.treeDepth)) * params.tree_count;

  // using random matrix to simulate training an RF classifer 
  getModelAndInput(numberOfPaths, params.dimension, modelParameter, negativeParameterMapping, compareParameterMapping);

  // generate random inputs
  vector<vector<int>> modelInput;
  modelInput.resize(numberOfPaths, vector<int>(params.dimension));
  vector<vector<int>> eqCompareParameterMapping; 
  eqCompareParameterMapping.resize(numberOfPaths, vector<int>(params.dimension));

  auto start = std::chrono::high_resolution_clock::now();
  
  #pragma omp parallel for
    for (uint64_t row = 0; row < numberOfPaths; ++row) {
      int rand_input = (row*row % 501) - 250;  // generate random input that should be the same across each row 
      for (uint64_t col = 0; col < params.dimension; ++col) {

        modelInput[row][col] = rand_input * negativeParameterMapping[row][col];  // will either be "rand_input * -1" or "rand_input * 1"
        if(compareParameterMapping[row][col] == LEQ or compareParameterMapping[row][col] == GEQ) {
          // if the operator uses an equal, set it to use the input against the model parameter in pim
          eqCompareParameterMapping[row][col] = modelInput[row][col];
        } else {
          // else, set the equal operator to something thats NOT the model parameter so it always evaluates to false 
          eqCompareParameterMapping[row][col] = modelParameter[row][col] << 1;  // bitshift since thats a fast operation
        }
      }
    }

  auto end = std::chrono::high_resolution_clock::now();
  chrono::duration<double, milli> inputTime = end - start;

  if (!createDevice(params.configFile))
    return 1;

  // load predictions into var
  vector<int> rfResult(numberOfPaths);
  int res = runRF(numberOfPaths, params.tree_count, params.dimension, modelInput, eqCompareParameterMapping, rfResult);


  pimShowStats();
  cout << "Host elapsed time for preprocessing: " << fixed << setprecision(3) << inputTime.count() << " ms." << endl;

  cout << "Host elapsed time: " << fixed << setprecision(3) << hostElapsedTime.count() << " ms." << endl;

  return 0;
}
