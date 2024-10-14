// Test: C++ version of vgg16
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
#include "../../util.h"
#include "../../utilML.h"
#include "../../utilFixedPoint.h"
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
          "\nUsage:  ./vgg16.out [options]"
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
  int KernelHeight = 3; 
  int kernelWidth = 3;
  int kernelDepth = 64;
  // Padding for the input image
  int padding = 1; 

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
      getMatrix(KernelHeight, kernelWidth, 0, mat);
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
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
  std::cout << "........ending conv1-1........\n";

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv1-2
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
  for (unsigned long int i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(224, 224, 1, resultMatrix1[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv1-2........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
  std::cout << "........ending conv1-2........\n";
 
  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // pool
  std::cout << "........starting pooling........\n";
  std::vector<std::vector<std::vector<int>>> resultMatrix2;
  pool(resultMatrix1, 2, 2, 2, resultMatrix2);
  std::cout << "........ending pooling........\n";

  // conv2-1
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
    kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.5.weight");	    
    if (params.shouldVerify == true) {
      kernelMatrix = floatToFixed(kernelMatrix_f);
    } else {
      kernelMatrix = binarizeMatrix(kernelMatrix_f);
    }  
  }  
  inputMatrix.clear();
  inputMatrix.resize(resultMatrix2.size());
  for (unsigned long int i = 0; i < resultMatrix2.size(); ++i)
  {
    addPadding(112, 112, 1, resultMatrix2[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv2-1........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
  std::cout << "........ending conv2-1........\n";

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv2-2
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
    kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.7.weight");	     
    if (params.shouldVerify == true) {
      kernelMatrix = floatToFixed(kernelMatrix_f);
    } else {
      kernelMatrix = binarizeMatrix(kernelMatrix_f);
    }  
  }  
  inputMatrix.clear();
  inputMatrix.resize(128);
  for (unsigned long int i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(112, 112, 1, resultMatrix1[i], inputMatrix[i]);
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

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // pool
  std::cout << "........starting pooling........\n";
  resultMatrix2.clear();
  pool(resultMatrix1, 2, 2, 2, resultMatrix2);
  std::cout << "........ending pooling........\n";

  // conv3-1
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
    kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.10.weight");    
    if (params.shouldVerify == true) {
      kernelMatrix = floatToFixed(kernelMatrix_f);
    } else {
      kernelMatrix = binarizeMatrix(kernelMatrix_f);
    }  
  }  
  inputMatrix.clear();
  inputMatrix.resize(128);
  for (unsigned long int i = 0; i < resultMatrix2.size(); ++i)
  {
    addPadding(56, 56, 1, resultMatrix2[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv3-1........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);       
  }    
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
  std::cout << "........ending conv3-1........\n";

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv3-2
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
    kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.12.weight");	      
    if (params.shouldVerify == true) {
      kernelMatrix = floatToFixed(kernelMatrix_f);
    } else {
      kernelMatrix = binarizeMatrix(kernelMatrix_f);
    }  
  }  
  inputMatrix.clear();
  inputMatrix.resize(256);
  for (unsigned long int i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(56, 56, 1, resultMatrix1[i], inputMatrix[i]);
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

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv3-3
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
    kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.14.weight");	      
    if (params.shouldVerify == true) {
      kernelMatrix = floatToFixed(kernelMatrix_f);
    } else {
      kernelMatrix = binarizeMatrix(kernelMatrix_f);
    }  
  }  
  inputMatrix.clear();
  inputMatrix.resize(256);
  for (unsigned long int i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(56, 56, 1, resultMatrix1[i], inputMatrix[i]);
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
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
  std::cout << "........ending conv3-3........\n";

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // pool
  resultMatrix2.clear();
  std::cout << "........starting pooling........\n";
  pool(resultMatrix1, 2, 2, 2, resultMatrix2);
  std::cout << "........ending pooling........\n";

  // conv4-1
  kernelMatrix.clear();
  if (params.kernelMatrixFile == nullptr)
  {
    kernelMatrix.resize(512);
    for (auto &mat : kernelMatrix)
    {
      getMatrix(3, 3, 0, mat);
    }
  }  else
  {
    kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.17.weight");	      
    if (params.shouldVerify == true) {
      kernelMatrix = floatToFixed(kernelMatrix_f);
    } else {
      kernelMatrix = binarizeMatrix(kernelMatrix_f);
    }  
  }
  inputMatrix.clear();
  inputMatrix.resize(256);
  for (unsigned long int i = 0; i < resultMatrix2.size(); ++i)
  {
    addPadding(28, 28, 1, resultMatrix2[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv4-1........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);    
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
  std::cout << "........ending conv4-1........\n";

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv4-2
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
    kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.19.weight");	      
    if (params.shouldVerify == true) {
      kernelMatrix = floatToFixed(kernelMatrix_f);
    } else {
      kernelMatrix = binarizeMatrix(kernelMatrix_f);
    }  
  }  
  inputMatrix.clear();
  inputMatrix.resize(512);
  for (unsigned long int i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(28, 28, 1, resultMatrix1[i], inputMatrix[i]);
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

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv4-3
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
    kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.21.weight");	      
    if (params.shouldVerify == true) {
      kernelMatrix = floatToFixed(kernelMatrix_f);
    } else {
      kernelMatrix = binarizeMatrix(kernelMatrix_f);
    }  
  }  
  inputMatrix.clear();
  inputMatrix.resize(512);
  for (unsigned long int i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(28, 28, 1, resultMatrix1[i], inputMatrix[i]);
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
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
  std::cout << "........ending conv4-3........\n";

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // pool
  resultMatrix2.clear();
  std::cout << "........starting pooling........\n";
  pool(resultMatrix1, 2, 2, 2, resultMatrix2);
  std::cout << "........ending pooling........\n";

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
    kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.24.weight");	      
    if (params.shouldVerify == true) {
      kernelMatrix = floatToFixed(kernelMatrix_f);
    } else {
      kernelMatrix = binarizeMatrix(kernelMatrix_f);
    }  
  }  
  inputMatrix.clear();
  inputMatrix.resize(512);
  for (unsigned long int i = 0; i < resultMatrix2.size(); ++i)
  {
    addPadding(14, 14, 1, resultMatrix2[i], inputMatrix[i]);
  }
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv5-1........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and kernel matrices  
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel matrix dimensions: ";
    printMatrixDimensions(kernelMatrix);      
  }  
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
  std::cout << "........ending conv5-1........\n";

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv5-2
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
    kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.26.weight");	      
    if (params.shouldVerify == true) {
      kernelMatrix = floatToFixed(kernelMatrix_f);
    } else {
      kernelMatrix = binarizeMatrix(kernelMatrix_f);
    }  
  }  
  inputMatrix.clear();
  inputMatrix.resize(512);
  for (unsigned long int i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(14, 14, 1, resultMatrix1[i], inputMatrix[i]);
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

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv5-3
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
    kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.28.weight");	      
    if (params.shouldVerify == true) {
      kernelMatrix = floatToFixed(kernelMatrix_f);
    } else {
      kernelMatrix = binarizeMatrix(kernelMatrix_f);
    }  
  }  
  inputMatrix.clear();
  inputMatrix.resize(512);
  for (unsigned long int i = 0; i < resultMatrix1.size(); ++i)
  {
    addPadding(14, 14, 1, resultMatrix1[i], inputMatrix[i]);
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
  conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
  std::cout << "........ending conv5-3........\n";

  // RELU
  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // pool
  resultMatrix2.clear();
  std::cout << "........starting pooling........\n";
  pool(resultMatrix1, 2, 2, 2, resultMatrix2);
  std::cout << "........ending pooling........\n";

  // dense layer 1
  std::vector<int> flattenedMat;
  flatten3DMat(resultMatrix2, flattenedMat);
  std::vector<std::vector<float>> denseWeight_f;
  std::vector<std::vector<int>> denseWeight;
  std::vector<int> denseOutput1;
  if (params.kernelMatrixFile == nullptr)
  {
    getMatrix(25088, 4096, 0, denseWeight);
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
  gemv(4096, 25088, flattenedMat, denseWeight, denseOutput1);
  std::cout << "........ending dense1........\n";

  // RELU
  std::cout << "........starting RELU........\n";
  performRelu(denseOutput1);
  std::cout << "........ending RELU........\n";

  // dense layer 2
  denseWeight.clear();
  std::vector<int> denseOutput2;
  if (params.kernelMatrixFile == nullptr)
  {
    getMatrix(4096, 4096, 0, denseWeight);
  }
  else
  {
    denseWeight_f = read_dense_layer_weights_from_csv(params.kernelMatrixFile, "classifier.3.weight");	      
    if (params.shouldVerify == true) {
      denseWeight = floatToFixed(denseWeight_f);
    } else {
      denseWeight = binarizeMatrix(denseWeight_f);
    }  
  }  
  std::cout << "........starting dense2........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and weight matrices  
    std::cout << "Input matrix dimensions: " << denseOutput1.size() << std::endl;
    std::cout << "Weight matrix dimensions: ";
    printMatrixDimensions(denseWeight);      
  }  
  gemv(4096, 4096, denseOutput1, denseWeight, denseOutput2);
  std::cout << "........ending dense2........\n";

  // RELU
  std::cout << "........starting RELU........\n";
  performRelu(denseOutput2);
  std::cout << "........ending RELU........\n";

  // dense layer 3
  denseWeight.clear();
  std::vector<int> denseOutput3;
  if (params.kernelMatrixFile == nullptr)
  {
    getMatrix(4096, 1000, 0, denseWeight);
  }
  else
  {
    denseWeight_f = read_dense_layer_weights_from_csv(params.kernelMatrixFile, "classifier.6.weight");	      
    if (params.shouldVerify == true) {
      denseWeight = floatToFixed(denseWeight_f);
    } else {
      denseWeight = binarizeMatrix(denseWeight_f);
    }      
  }  
  std::cout << "........starting dense3........\n";
  if (params.moreDebugPrints == true) { 
    // Check the dimensions of the input and weight matrices  
    std::cout << "Input matrix dimensions: " << denseOutput2.size() << std::endl;
    std::cout << "Weight matrix dimensions: ";
    printMatrixDimensions(denseWeight);     
  }  
  gemv(1000, 4096, denseOutput2, denseWeight, denseOutput3);
  std::cout << "........ending dense3........\n";

  // perform softmax in host
  std::vector<double> resultVector;  
  std::vector<float> denseOutput3_f;
  denseOutput3_f = fixedToFloat(denseOutput3); 
  auto start = std::chrono::high_resolution_clock::now();
  if (params.shouldVerify == true) {
    softmaxOnHost(denseOutput3_f, resultVector); 
  } else {
    softmaxOnHost(denseOutput3, resultVector);
  }
  auto end = std::chrono::high_resolution_clock::now();
  hostElapsedTime += (end - start);
  std::cout << "Dimensions of the softmax output: " << resultVector.size() <<  std::endl;

  // *********************************************************************************************************************************
  //  Verification Process:
  //  The following code initializes a vector of pairs, fills it with the softmax output values and their corresponding indices.
  //  Then sorts the pairs by value in descending order, and prints out the top 5 values along with their indices. 
  //  These indices can be used to map back to the original 1000 output classes of the VGG16 model.  
  //  The top 5 results are printed as the goal is to see if the correct class label is among the indices of the top 5 values.
  //  This is important in many classification tasks where the top prediction might not always be correct, 
  //  but the correct label might still be within the top 5 highest probability predictions.    
  // *********************************************************************************************************************************

  // Create a vector of pairs to store value-index pairs
  std::vector<std::pair<double, int>> valueIndexPairs;
  // Populate the vector with value-index pairs
  for (unsigned long int i = 0; i < resultVector.size(); ++i) {
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
