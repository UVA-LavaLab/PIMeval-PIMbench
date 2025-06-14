// Test: C++ version of ResNet18
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <getopt.h>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "util.h"
#include "utilML.h"
#include "utilFixedPoint.h"
#include <iomanip>
#include <chrono>
#include <cassert>

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  char *dramConfigFile;
  char *imageInputFile;
  char *kernelMatrixFile;
  bool shouldVerify;
  bool moreDebugPrints;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./resnet18.out [options]"
          "\n"
          "\n    -v    should verify result with CPU"
          "\n    -i    input image file (default=generates matrix with random numbers)"
          "\n    -k    input csv file containing the kernel matrices (default=generates matrices with random numbers)"
          "\n    -c    input file containing dramsim config"
          "\n    -m    enable more debug prints (default = false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.dramConfigFile = nullptr;
  p.imageInputFile = nullptr;
  p.kernelMatrixFile = nullptr;
  p.shouldVerify = false;
  p.moreDebugPrints = false;

  int opt;
  while ((opt = getopt(argc, argv, "c:v:i:k:m:")) >= 0)
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
    case 'i':
      p.imageInputFile = optarg;
      break;
    case 'k':
      p.kernelMatrixFile = optarg;
      break;  
    case 'v':
      p.shouldVerify = (*optarg == 't') ? true : false;
      break;
    case 'm':
      p.moreDebugPrints = (*optarg == 't') ? true : false; 
      break;       
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

void convolutionBlock(std::vector<std::vector<std::vector<int>>> &inputMatrix) {
  return;
}

void identityBlock() {
  return;
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::vector<std::vector<std::vector<float>>> inputMatrix_f;
  std::vector<std::vector<std::vector<float>>> kernelMatrix_f;
  std::vector<std::vector<std::vector<int>>> inputMatrix;
  std::vector<std::vector<std::vector<int>>> kernelMatrix;
  // Dimensions of the input image
  int imageHeight = 224;
  int imageWidth = 224;
  int imageDepth = 3;
  // Dimensions of the kernel in the first convolutional layer
  int kernelHeight = 7; 
  int kernelWidth = 7;
  int kernelDepth = 64;
  // Padding for the input image
  int padding = 3; 

  if (params.imageInputFile == nullptr)
  {
    inputMatrix.resize(imageDepth);
    for (int i = 0; i < imageDepth; i++)
    {
      getMatrix(imageHeight, imageWidth, padding, inputMatrix[i]);
    }
  }
  else // Get inputMatrix from the input image
  {
    #ifdef COMPILE_WITH_JPEG 
    // If JPEG lib is not supported, below code will not work. In that case disable JPEG by adding COMPILE_WITH_JPEG=0 during make. 
    std::string outputFile = "resized_output.jpg";
    // Matrix to store resized image data
    std::vector<std::vector<std::vector<int>>> inputMatrixBeforePadding;
    // Resize the input JPEG image
    readJPEG(params.imageInputFile, inputMatrixBeforePadding, imageHeight, imageWidth);
    // Successfully resized image, now write to output JPEG file
    writeResizedImage(outputFile, inputMatrixBeforePadding);
    // Padding the resized input image
    int depth = inputMatrixBeforePadding.size();
    if (depth != imageDepth) {
      std::cerr << "Assertion failed: depth (" << depth << ") != imageDepth (" << imageDepth << ")\n";
      assert(depth == imageDepth && "Given input image depth does not match with the expected image depth");  
    }    
    inputMatrix.resize(depth); 
    for (int d = 0; d < depth; ++d) {
      addPadding(imageHeight, imageWidth, padding, inputMatrixBeforePadding[d], inputMatrix[d]);
    }
    #endif  
  }
  if (params.kernelMatrixFile == nullptr)
  {
    kernelMatrix.resize(kernelDepth);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(kernelHeight, kernelWidth, 0, mat);
    }
  }
  else
  {
    kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.0.weight");	  
    if (params.shouldVerify == true) {
      kernelMatrix = floatToFixed(kernelMatrix_f);
    } else {
      kernelMatrix = binarizeMatrix(kernelMatrix_f);
    }  
  }

  if (!createDevice(params.dramConfigFile))
    return 1;
    
  // conv1-1
  std::cout << "........starting conv1-1........\n";
  std::vector<std::vector<std::vector<int>>> resultMatrix1;
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix); 
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 2, 1);
  std::cout << "........ending conv1-1........\n";
  if (params.moreDebugPrints == true) { 
    printMatrixDimensions(resultMatrix1);
  }

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // max pool
  std::vector<std::vector<std::vector<int>>> pooledMatrix;
  std::cout << "........starting max pool........\n";
  pool(resultMatrix1, 3, 3, 2, pooledMatrix);
  resultMatrix1 = pooledMatrix;
  std::cout << "........ending max pool........\n";
  if (params.moreDebugPrints == true) { 
    printMatrixDimensions(resultMatrix1);
  }

  // conv2-1
  kernelMatrix.clear();
  if (params.kernelMatrixFile == nullptr)
  {
    kernelMatrix.resize(64);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(3, 3, 0, mat);
    }
  }
  else
  {
    kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.2.weight");	  
    if (params.shouldVerify == true) {
      kernelMatrix = floatToFixed(kernelMatrix_f);
    } else {
      kernelMatrix = binarizeMatrix(kernelMatrix_f);
    }  
  }
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix1.size());
  std::vector<std::vector<std::vector<int>>> skipInput;
  skipInput.resize(resultMatrix1.size());
  for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(56, 56, 0, resultMatrix1[i], skipInput[i]);
    addPadding(56, 56, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();

  std::vector<std::vector<std::vector<int>>> skipResult;
  skipResult.clear();
  skipResult.shrink_to_fit();
  std::vector<std::vector<std::vector<int>>> skipKernel;
  skipKernel.resize(64);
  for (auto &mat : skipKernel)
  {
    getMatrix(1, 1, 0, mat);
  }
  std::cout << "........starting conv2-1........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 0);
  std::cout << "........ending conv2-1........\n";
  if (params.moreDebugPrints == true) { 
    printMatrixDimensions(resultMatrix1);
  }

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv2-forward
  std::cout << "........starting forward conv........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(skipInput);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(skipKernel);    
  }  
  conv2(skipInput, skipKernel, skipResult, 1, 1);
  std::cout << "........ending forward conv........\n";
  if (params.moreDebugPrints == true) {
    printMatrixDimensions(skipResult);
  }  

  // conv2-2
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix1.size());
  for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(56, 56, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv2-2........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
  std::cout << "........ending conv2-2........\n";
  if (params.moreDebugPrints == true) { 
    printMatrixDimensions(resultMatrix1);
  }

  // Addition phase for convolution block
  std::vector<std::vector<std::vector<int>>> addResult;
  addResult.resize(resultMatrix1.size());
  addMatrices(resultMatrix1, skipResult, addResult);
  resultMatrix1 = addResult;

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv2-3
  skipResult = resultMatrix1;
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix1.size());
  for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(56, 56, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv2-3........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 0);
  std::cout << "........ending conv2-3........\n";
  if (params.moreDebugPrints == true) { 
    printMatrixDimensions(resultMatrix1);
  }

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv2-4
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix1.size());
  for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(56, 56, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv2-4........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 0);
  std::cout << "........ending conv2-4........\n";
  if (params.moreDebugPrints == true) { 
    printMatrixDimensions(resultMatrix1);
  }

  // RELU for identity
  std::cout << "........starting RELU for identity........\n";
  relu(skipResult);
  std::cout << "........ending RELU for identity........\n";

  // Addition phase for identity block
  addResult.resize(resultMatrix1.size());
  addMatrices(resultMatrix1, skipResult, addResult);
  resultMatrix1 = addResult;

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv3-1
  kernelMatrix.clear();
  if (params.kernelMatrixFile == nullptr)
  {
    kernelMatrix.resize(128);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(3, 3, 0, mat);
    }
  }
  else
  {
    kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.2.weight");	  
    if (params.shouldVerify == true) {
      kernelMatrix = floatToFixed(kernelMatrix_f);
    } else {
      kernelMatrix = binarizeMatrix(kernelMatrix_f);
    }  
  }
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix1.size());
  skipInput.resize(resultMatrix1.size());
  for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(56, 56, 0, resultMatrix1[i], skipInput[i]);
    addPadding(56, 56, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();

  skipResult.clear();
  skipResult.shrink_to_fit();
  skipKernel.resize(128);
  for (auto &mat : skipKernel)
  {
    getMatrix(1, 1, 0, mat);
  }

  std::cout << "........starting conv3-1........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 2, 1);
  std::cout << "........ending conv3-1........\n";
  if (params.moreDebugPrints == true) {
    printMatrixDimensions(resultMatrix1);
  }

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv3-forward
  std::cout << "........starting forward conv........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(skipInput);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(skipKernel);    
  }  
  conv2(skipInput, skipKernel, skipResult, 2, 1);
  std::cout << "........ending forward conv........\n";
  if (params.moreDebugPrints == true) {
    printMatrixDimensions(skipResult);
  }

  // conv3-2
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix1.size());
  for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(28, 28, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv3-2........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
  std::cout << "........ending conv3-2........\n";
  if (params.moreDebugPrints == true) {
    printMatrixDimensions(resultMatrix1);
  }

  // Addition phase for convolution block
  addResult.resize(resultMatrix1.size());
  addMatrices(resultMatrix1, skipResult, addResult);
  resultMatrix1 = addResult;

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv3-3
  skipResult = resultMatrix1;
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix1.size());
  for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(28, 28, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv3-3........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 0);
  std::cout << "........ending conv3-3........\n";
  if (params.moreDebugPrints == true) { 
    printMatrixDimensions(resultMatrix1);
  }

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv3-4
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix1.size());
  for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(28, 28, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv3-4........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 0);
  std::cout << "........ending conv3-4........\n";
  if (params.moreDebugPrints == true) { 
    printMatrixDimensions(resultMatrix1);
  }

  // RELU for identity
  std::cout << "........starting RELU for identity........\n";
  relu(skipResult);
  std::cout << "........ending RELU for identity........\n";

  // Addition phase for identity block
  addResult.resize(resultMatrix1.size());
  addMatrices(resultMatrix1, skipResult, addResult);
  resultMatrix1 = addResult;

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv4-1
  kernelMatrix.clear();
  if (params.kernelMatrixFile == nullptr)
  {
    kernelMatrix.resize(256);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(3, 3, 0, mat);
    }
  }
  else
  {
    kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.2.weight");	  
    if (params.shouldVerify == true) {
      kernelMatrix = floatToFixed(kernelMatrix_f);
    } else {
      kernelMatrix = binarizeMatrix(kernelMatrix_f);
    }  
  }
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix1.size());
  skipInput.clear();
  skipInput.resize(resultMatrix1.size());
  for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(28, 28, 0, resultMatrix1[i], skipInput[i]);
    addPadding(28, 28, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  skipResult.clear();
  skipResult.shrink_to_fit();
  skipKernel.resize(256);
  for (auto &mat : skipKernel)
  {
    getMatrix(1, 1, 0, mat);
  }

  std::cout << "........starting conv4-1........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 2, 1);
  std::cout << "........ending conv4-1........\n";
  if (params.moreDebugPrints == true) {
    printMatrixDimensions(resultMatrix1);
  }

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv4-forward
  std::cout << "........starting forward conv........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(skipInput);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(skipKernel);    
  }  
  conv2(skipInput, skipKernel, skipResult, 2, 1);
  std::cout << "........ending forward conv........\n";
  if (params.moreDebugPrints == true) {
    printMatrixDimensions(skipResult);
  }

  // conv4-2
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix1.size());
  for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(14, 14, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv4-2........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
  std::cout << "........ending conv4-2........\n";
  if (params.moreDebugPrints == true) {
    printMatrixDimensions(resultMatrix1);
  }

  // Addition phase for convolution block
  addResult.resize(resultMatrix1.size());
  addMatrices(resultMatrix1, skipResult, addResult);
  resultMatrix1 = addResult;

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv4-3
  skipResult = resultMatrix1;
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix1.size());
  for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(14, 14, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv4-3........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 0);
  std::cout << "........ending conv4-3........\n";
  if (params.moreDebugPrints == true) { 
    printMatrixDimensions(resultMatrix1);
  }

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv4-4
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix1.size());
  for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(14, 14, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv4-4........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 0);
  std::cout << "........ending conv4-4........\n";
  if (params.moreDebugPrints == true) { 
    printMatrixDimensions(resultMatrix1);
  }

  // RELU for identity
  std::cout << "........starting RELU for identity........\n";
  relu(skipResult);
  std::cout << "........ending RELU for identity........\n";

  // Addition phase for identity block
  addResult.resize(resultMatrix1.size());
  addMatrices(resultMatrix1, skipResult, addResult);
  resultMatrix1 = addResult;

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv5-1
  kernelMatrix.clear();
  if (params.kernelMatrixFile == nullptr)
  {
    kernelMatrix.resize(512);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(3, 3, 0, mat);
    }
  }
  else
  {
    kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.2.weight");	  
    if (params.shouldVerify == true) {
      kernelMatrix = floatToFixed(kernelMatrix_f);
    } else {
      kernelMatrix = binarizeMatrix(kernelMatrix_f);
    }  
  }
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix1.size());
  skipInput.clear();
  skipInput.resize(resultMatrix1.size());
  for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(14, 14, 0, resultMatrix1[i], skipInput[i]);
    addPadding(14, 14, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  skipResult.clear();
  skipResult.shrink_to_fit();
  skipKernel.resize(512);
  for (auto &mat : skipKernel)
  {
    getMatrix(1, 1, 0, mat);
  }

  std::cout << "........starting conv5-1........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 2, 1);
  std::cout << "........ending conv5-1........\n";
  if (params.moreDebugPrints == true) {
    printMatrixDimensions(resultMatrix1);
  }

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv5-forward
  std::cout << "........starting forward conv........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(skipInput);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(skipKernel);    
  }  
  conv2(skipInput, skipKernel, skipResult, 2, 1);
  std::cout << "........ending forward conv........\n";
  if (params.moreDebugPrints == true) {
    printMatrixDimensions(skipResult);
  }

  // conv5-2
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix1.size());
  for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(7, 7, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv5-2........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
  std::cout << "........ending conv5-2........\n";
  if (params.moreDebugPrints == true) {
    printMatrixDimensions(resultMatrix1);
  }

  // Addition phase for convolution block
  addResult.resize(resultMatrix1.size());
  addMatrices(resultMatrix1, skipResult, addResult);
  resultMatrix1 = addResult;

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv5-3
  skipResult = resultMatrix1;
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix1.size());
  for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(7, 7, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv5-3........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 0);
  std::cout << "........ending conv5-3........\n";
  if (params.moreDebugPrints == true) { 
    printMatrixDimensions(resultMatrix1);
  }

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv5-4
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix1.size());
  for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(7, 7, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv5-4........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 0);
  std::cout << "........ending conv5-4........\n";
  if (params.moreDebugPrints == true) { 
    printMatrixDimensions(resultMatrix1);
  }

  // RELU for identity
  std::cout << "........starting RELU for identity........\n";
  relu(skipResult);
  std::cout << "........ending RELU for identity........\n";

  // Addition phase for identity block
  addResult.resize(resultMatrix1.size());
  addMatrices(resultMatrix1, skipResult, addResult);
  resultMatrix1 = addResult;

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // max pool
  pooledMatrix.clear();
  std::cout << "........starting max pool........\n";
  pool(resultMatrix1, 2, 2, 2, pooledMatrix);
  std::cout << "........ending max pool........\n";
  resultMatrix1 = pooledMatrix;

  // dense layer
  std::vector<int> flattenedMat;
  flatten3DMat(resultMatrix1, flattenedMat);
  std::vector<std::vector<float>> denseWeight_f;
  std::vector<std::vector<int>> denseWeight;
  std::vector<int> denseOutput;
  if (params.kernelMatrixFile == nullptr)
  {
    getMatrix(512, 1000, 0, denseWeight);
  }
  else
  {
    denseWeight_f = read_dense_layer_weights_from_csv(params.kernelMatrixFile, "classifier.0.weight");	      
    if (params.shouldVerify == true) {
      denseWeight = floatToFixed(denseWeight_f);
    } else {
      denseWeight = binarizeMatrix(denseWeight_f);
    }  
  }  
  std::cout << "........starting dense1........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and weight matrices  
    std::cout << "Input matrix dimensions: " << flattenedMat.size() << std::endl;
    std::cout << "Weight matrix dimensions: ";
    printMatrixDimensions(denseWeight);       
  }   
  gemv(1000, 512, flattenedMat, denseWeight, denseOutput);
  std::cout << "........ending dense1........\n";

  // RELU
  std::cout << "........starting RELU........\n";
  performRelu(denseOutput);
  std::cout << "........ending RELU........\n";

  // perform softmax in host
  std::vector<double> resultVector;  
  std::vector<float> denseOutput_f;
  denseOutput_f = fixedToFloat(denseOutput); 
  auto start = std::chrono::high_resolution_clock::now();
  if (params.shouldVerify == true) {
    softmaxOnHost(denseOutput_f, resultVector); 
  } else {
    softmaxOnHost(denseOutput, resultVector);
  }
  auto end = std::chrono::high_resolution_clock::now();
  hostElapsedTime += (end - start);
  std::cout << "Dimensions of the softmax output: " << resultVector.size() <<  std::endl;  

  // Verification -- Taken entirely from the previous VGG implementations
  std::vector<std::pair<double, int>> valueIndexPairs;
  // Populate the vector with value-index pairs
  for (uint64_t i = 0; i < resultVector.size(); ++i) {
      valueIndexPairs.push_back(std::make_pair(resultVector[i], i));
  }
  // Sort the vector of pairs based on values in descending order
  std::sort(valueIndexPairs.begin(), valueIndexPairs.end(), std::greater<std::pair<double, int>>());
  // Print the top 5 values along with their indices
  std::cout << "Top 5 values and corresponding indices:\n";
  for (int i = 0; i < 5; ++i) {
      std::cout << "Value: " << valueIndexPairs[i].first << " Index: " << valueIndexPairs[i].second << std::endl;
  }

  pimShowStats();
  cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << endl;

  return 0;
}
