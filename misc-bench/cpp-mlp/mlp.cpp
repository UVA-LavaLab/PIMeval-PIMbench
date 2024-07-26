
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <stdint.h>

#include "../util.h"
#include "libpimeval.h"
using namespace std;

// Data type
#define T int32_t

typedef struct Params {
  vector<int> layer_sizes;
  int   num_layers;
  int   num_examples;
  bool shouldVerify;
  char *configFile;
}Params;

void usage() {
  fprintf(stderr,
    "\nUsage:  ./mlp.out [options]"
    "\n"
    "\n    -l    comma-seperate list of number the neurons in each layer. First number represents the input size, last represents output "
    "\n          size, and every number in between is the size of each hidden layer. (default=500 input neurons, 1 hidden layer with 1000 neurons, 2 output neurons)"    
    "\n    -c    dramsim config file"
    "\n    -n    number of examples (default=1)"
    "\n    -v    t = verifies PIM output with host output. (default=false)"
    "\n"
    "\nGeneral options:"
    "\n    -h        help"
    "\n"
    "\n");
}

void parse_layer_sizes(const string arg, Params &p) {
  istringstream ss(arg);
  string token;
  int count = 0;
  vector<int> temp_layer_sizes;

  while (getline(ss, token, ',')) {
    int sizeToken = stoi(token);
    if(sizeToken <= 0) {
      fprintf(stderr, "Layer can not have 0 or less neurons! Exiting.\n");
      exit(1);
    }
    temp_layer_sizes.push_back(sizeToken);
    count++;
  }
  if(count < 3) {
    fprintf(stderr, "Invalid number of layers!\n");
    exit(1);
  }
  p.layer_sizes = temp_layer_sizes;
  p.num_layers = count;
}

Params input_params(int argc, char **argv) {
  Params p;
  p.configFile = nullptr;
  p.layer_sizes = {500, 1000, 2};
  p.num_layers = 3;
  p.num_examples = 1;
  p.shouldVerify = false;

  int opt;
  while((opt = getopt(argc, argv, "h:l:n:c:v:")) >= 0) {
    switch(opt) {
      case 'h':
      usage();
      exit(0);
      break;
      case 'l':
        p.layer_sizes.clear();
        parse_layer_sizes(optarg, p);
        break;
      case 'n':
        p.num_examples = atoll(optarg);
        break;
      case 'v':
        p.shouldVerify = (*optarg == 't') ? true : false;
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

// Create input arrays
static void init_weights(vector<vector<vector<T>>>& weight, vector<int>& layers, unsigned int num_layers){

  // correctly size the weight matrix between each layer
  for (unsigned int nl = 0; nl < num_layers - 1; nl++) {
    unsigned int layerInputSize = layers[nl];
    unsigned int layerOutputSize = layers[nl + 1];
    weight[nl].resize(layerInputSize);

    for (unsigned int m = 0; m < layerInputSize; m++) {
      weight[nl][m].resize(layerOutputSize);
      for (unsigned int n = 0; n < layerOutputSize; n++){
        weight[nl][m][n] = rand() % 3 - 1;  // random weight value to put into the flattened weight matrix, either -1,0,1
      }
    }
  }
}

void gemv(uint64_t row, uint64_t col, std::vector<int> &srcVector, std::vector<std::vector<int>> &srcMatrix, std::vector<int> &dst)
{
  unsigned bitsPerElement = sizeof(int) * 8;
  PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, row, bitsPerElement, PIM_INT32);
  if (srcObj1 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
  PimObjId srcObj2 = pimAllocAssociated(bitsPerElement, srcObj1, PIM_INT32);
  if (srcObj2 == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId dstObj = pimAllocAssociated(bitsPerElement, srcObj1, PIM_INT32);
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
    // only if its the first run, otherwise it was provided by previous run
    status = pimCopyHostToDevice((void *)srcMatrix[i].data(), srcObj1);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimScaledAdd(srcObj1, dstObj, dstObj, srcVector[i]);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }
  }
  
  // ReLU Neuron activation
  status = pimMaxScalar(dstObj, dstObj, 0);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
  }

  dst.resize(row);
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

static bool mlp_pim(vector<vector<vector<T>>>& weight, vector<vector<T>>& input, vector<vector<T>>& res, vector<T>& layers, int num_layers, bool shouldVerify) {
  vector<vector<vector<T>>> inputOutputLayerT(num_layers - 1);
  vector<vector<vector<T>>> weightLayerT(num_layers - 1);
  int numExamples = input.size();
  // allocate transposed weight and input/Output layer matricies
  for(int n = 0; n < num_layers - 1; n++) {
    int weightRow = weight[n].size();
    int weightCol = weight[n][0].size();

    inputOutputLayerT[n].resize(weightRow);
    for(int j = 0; j < weightRow; j++) {
      inputOutputLayerT[n][j].resize(numExamples);
    }

    weightLayerT[n].resize(weightCol);
    for(int j = 0; j < weightCol; j++) {
      weightLayerT[n][j].resize(weightRow);
    }
  }
  
  // intialize input into the first layer
  transposeMatrix(numExamples, weight[0].size(), input, inputOutputLayerT[0]);
  // GEMM - TODO: Do we actually need to transpose matrices
  for(int n = 0; n < num_layers - 1; n++){
    int weightRow = weight[n].size();
    int weightCol = weight[n][0].size();
    vector<vector<T>> tempLayer(weightCol, vector<T>(numExamples));
    if(n == 0) {
    }
    transposeMatrix(weightRow, weightCol, weight[n], weightLayerT[n]);
    for (int i = 0; i < weightCol; ++i)
    {
      gemv(numExamples, weightRow, weightLayerT[n][i], inputOutputLayerT[n], tempLayer[i]);
    }
    if(n+1 == num_layers -1) {
      // if this is the last hidden layer, update the result layer
      res = tempLayer;
    } else {
      // otherwise, feed forward to the next hidden layer
      inputOutputLayerT[n+1] = tempLayer;
    }

    if (shouldVerify)
    {
      std::vector<std::vector<T>> C(numExamples, std::vector<T>(weightCol));
      for (int i = 0; i < numExamples; ++i)
      {
        for (int j = 0; j < weightCol; ++j)
        {
          for (int k = 0; k < weightRow; ++k)
          {
            C[i][j] += inputOutputLayerT[n][k][i] * weightLayerT[n][j][k];
          }
          C[i][j] = max(C[i][j], 0);
        }
      }
      for (int i = 0; i < numExamples; ++i)
      {
        for (int j = 0; j < weightCol; ++j)
        {
          if (C[i][j] != tempLayer[j][i])
          {
            std::cout << "Error: Incorrect Result.\nHost: " << C[i][j] << "\t PIM: " << tempLayer[i][j] << "\n";
            return false;
          }
        }
      }
    }

  }

  return true;
}


int main(int argc, char **argv) {

  struct Params p = input_params(argc, argv);
  uint64_t num_layers = p.num_layers;  // number of hidden layers
  vector<int> layers = p.layer_sizes;
  vector<vector<vector<T>>> weight(num_layers - 1);  // store the weight of each layer in a flattend matrix format 

  vector<vector<T>> input;
  getMatrix(p.num_examples, layers[0], 0, input);
  vector<vector<T>> outputResult;

  // Create an input file with arbitrary data.
  init_weights(weight, layers, num_layers);
  if (!createDevice(p.configFile))
    return 1;

  bool correctPimResult = mlp_pim(weight, input, outputResult, layers, num_layers, p.shouldVerify);

  pimShowStats();
  
  if(p.shouldVerify && correctPimResult) {
    cout << "KNN was succesfully verified against host result\n";
  } else if(p.shouldVerify && !correctPimResult) {
    cerr << "MLP verification failed!\n";
  }

  return 0;
}
