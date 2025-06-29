// Test: C++ version of batch normalization
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <cmath>
#include <getopt.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "util.h"

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t batch_size, num_features, height, width;
  float eps;
  bool affine;
  char *dramConfigFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./batch_norm.out [options]"
          "\n"
          "\n    -b    batch size (default=64)"
          "\n    -f    number of features (default=64)"
          "\n    -r    height (default=224)"
          "\n    -c    width (default=224)"
          "\n    -e    epsilon value (default=1e-5)"
          "\n    -a    affine transformation (default=false)"
          "\n    -d    DRAM config file"
          "\n    -v    should verify result with CPU"
          "\n    -g    enable more debug prints (default = false)"          
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.batch_size = 64;
  p.num_features = 3;
  p.height = 224;
  p.width = 224;
  p.eps = 1e-4;
  p.affine = false;
  p.dramConfigFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "hb:f:r:c:e:a:d:v:g:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'b':
      p.batch_size = atoi(optarg);
      break;
    case 'f':
      p.num_features = atoi(optarg);
      break;
    case 'r':
      p.height = atoi(optarg);
      break;
    case 'c':
      p.width = atoi(optarg);
      break;
    case 'e':
      p.eps = atof(optarg);
      break;
    case 'a':
      p.affine = (*optarg == 't') ? true : false;
      break;
    case 'd':
      p.dramConfigFile = optarg;
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

// Generate random input tensor for batch normalization
void generateRandomInput(int batch_size, int num_features, int height, int width, 
                        std::vector<std::vector<std::vector<std::vector<float>>>> &input_tensor)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0, 1.0);
  
  input_tensor.resize(batch_size);
  for (int b = 0; b < batch_size; b++) {
    input_tensor[b].resize(num_features);
    for (int f = 0; f < num_features; f++) {
      input_tensor[b][f].resize(height);
      for (int h = 0; h < height; h++) {
        input_tensor[b][f][h].resize(width);
        for (int w = 0; w < width; w++) {
          input_tensor[b][f][h][w] = dist(gen);
        }
      }
    }
  }
}

// CPU reference implementation of batch normalization
void cpuBatchNorm(const std::vector<std::vector<std::vector<std::vector<float>>>> &input,
                  std::vector<std::vector<std::vector<std::vector<float>>>> &output,
                  int num_features, float eps, bool affine)
{
  auto cpu_start = std::chrono::high_resolution_clock::now();
   
  int batch_size = input.size();
  int height = input[0][0].size();
  int width = input[0][0][0].size();
  
  output.resize(batch_size);
  for (int b = 0; b < batch_size; b++) {
    output[b].resize(num_features);
    for (int f = 0; f < num_features; f++) {
      output[b][f].resize(height);
      for (int h = 0; h < height; h++) {
        output[b][f][h].resize(width);
      }
    }
  }
  
  // Initialize learnable parameters
  std::vector<float> gamma(num_features, 1.0f);
  std::vector<float> beta(num_features, 0.0f);
  
  if (affine) {
    // Initialize gamma and beta with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(1.0, 0.1);
    for (int f = 0; f < num_features; f++) {
      gamma[f] = dist(gen);
      beta[f] = dist(gen);
    }
  }
  auto cpu_end = std::chrono::high_resolution_clock::now();
  auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
   std::cout << "[INFO] CPU computation time: " << cpu_duration.count() / 1000.0 << " ms" << std::endl;
    
    
  // For each feature channel
  for (int f = 0; f < num_features; f++) {
    // Compute mean and variance for this batch and feature
    float sum = 0.0f;
    float sum_sq = 0.0f;
    int count = batch_size * height * width;
    
    for (int b = 0; b < batch_size; b++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          float val = input[b][f][h][w];
          sum += val;
          sum_sq += val * val;
        }
      }
    }
    float mean = sum / count;
    float var = (sum_sq / count) - (mean * mean);
    var = std::max(var, 0.0f); // Ensure variance is non-negative
    // Normalize
    float std_dev = std::sqrt(var + eps);

    std::cout << "Feature " << f << " Mean: " << mean << ", Std Dev: " << std_dev << std::endl;
    
    for (int b = 0; b < batch_size; b++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          float normalized = (input[b][f][h][w] - mean) / std_dev;
          if (affine){
            output[b][f][h][w] = gamma[f] * normalized + beta[f];
          }
        }
      }
    }
    
  }
}

// PIM implementation of batch normalization
void pimBatchNorm(const std::vector<std::vector<std::vector<std::vector<float>>>> &input,
                  std::vector<std::vector<std::vector<std::vector<float>>>> &output,
                  int num_features, float eps, bool affine) {
    std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();
    auto start_cpu = std::chrono::high_resolution_clock::now();
    int batch_size = input.size();
    int height = input[0][0].size();
    int width = input[0][0][0].size();

    // Allocate output tensor shape
    output.resize(batch_size);
    for (int b = 0; b < batch_size; b++) {
        output[b].resize(num_features);
        for (int f = 0; f < num_features; f++) {
            output[b][f].resize(height);
            for (int h = 0; h < height; h++) {
                output[b][f][h].resize(width);
            }
        }
    }

    // Initialize affine parameters
    std::vector<float> gamma(num_features, 1.0f);
    std::vector<float> beta(num_features, 0.0f);
    if (affine) {
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(1.0f, 0.1f);
        for (int f = 0; f < num_features; ++f) {
            gamma[f] = dist(gen);
            beta[f] = dist(gen);
        }
    }
    auto stop_cpu = std::chrono::high_resolution_clock::now();
    hostElapsedTime += (stop_cpu - start_cpu);
    std::cout << "Host elapsed time: " << fixed << setprecision(3)<< hostElapsedTime.count() << " ms" << std::endl;

    for (int f = 0; f < num_features; ++f) {
        int data_size = batch_size * height * width;
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        for (int b = 0; b < batch_size; ++b){
          for (int h = 0; h < height; ++h){

            PimObjId input_obj = pimAlloc(PIM_ALLOC_AUTO, width, PIM_FP32);
            if (input_obj == -1) {
                std::cout << "Abort: pimAlloc failed for input_obj" << std::endl;
                return;
            }

            PimStatus status = pimCopyHostToDevice((void*)input[b][f][h].data(), input_obj);
            if (status != PIM_OK) {
                std::cout << "Abort" << std::endl;
                return;
            }

            float local_sum = 0.0f;
            status = pimRedSum(input_obj, &local_sum);
            if (status != PIM_OK)
            {
              std::cout << "Abort" << std::endl;
              return;
            }


            float local_sum_sq = 0.0f;
            status = pimMul(input_obj, input_obj, input_obj);
            if (status != PIM_OK)
            {
              std::cout << "Abort" << std::endl;
              return;
            }
            status = pimRedSum(input_obj, &local_sum_sq);
            if (status != PIM_OK){
              std::cout << "Abort" << std::endl;
              return;
            }

            sum += local_sum;
            sum_sq += local_sum_sq;

            pimFree(input_obj);

          }
        }

        float mean = sum / data_size;
        float var = (sum_sq / data_size) - (mean * mean);
        var = std::max(var, 0.0f); // Ensure variance is non-negative
        float std_dev = std::sqrt(var + eps);
        std::cout << "Feature " << f << " Mean: " << mean << ", Std Dev: " << std_dev << std::endl;
       
        for (int b = 0; b < batch_size; ++b){
          for (int h = 0; h < height; ++h){

            PimObjId output_obj = pimAlloc(PIM_ALLOC_AUTO, width, PIM_FP32);
            if (output_obj == -1) {
                std::cout << "Abort: pimAlloc failed for output_obj" << std::endl;
                return;
            }

            PimStatus status = pimCopyHostToDevice((void*)input[b][f][h].data(), output_obj);
            if (status != PIM_OK) {
                std::cout << "Abort" << std::endl;
                return;
            }

            PimObjId temp_obj = pimAllocAssociated(output_obj, PIM_FP32);
            if (temp_obj == -1) {
                std::cout << "Abort: pimAlloc failed for temp_obj" << std::endl;
                return;
            }


            // Normalize
            status = pimBroadcastFP(temp_obj, mean);
            if (status != PIM_OK) {
                std::cout << "Abort" << std::endl;
                return;
            }
            status = pimSub(output_obj, temp_obj, output_obj);
            if (status != PIM_OK) {
                std::cout << "Abort" << std::endl;
                return;
            }

            status = pimBroadcastFP(temp_obj, std_dev);
            if (status != PIM_OK) {
                std::cout << "Abort" << std::endl;
                return;
            }
            status = pimDiv(output_obj, temp_obj, output_obj);
            if (status != PIM_OK) {
                std::cout << "Abort" << std::endl;
                return;
            }

            // Apply affine transformation
            if (affine) {

                status = pimBroadcastFP(temp_obj, gamma[f]);
                if (status != PIM_OK) {
                    std::cout << "Abort" << std::endl;
                    return;
                }
                status = pimMul(output_obj, temp_obj, output_obj);
                if (status != PIM_OK) {
                    std::cout << "Abort" << std::endl;
                    return;
                }

                status = pimBroadcastFP(temp_obj, beta[f]);
                if (status != PIM_OK) {
                    std::cout << "Abort" << std::endl;
                    return;
                }
                status = pimAdd(output_obj, temp_obj, output_obj);
                if (status != PIM_OK) {
                    std::cout << "Abort" << std::endl;
                    return;
                }
            }
            
            status = pimCopyDeviceToHost(output_obj, output[b][f][h].data());
            if (status != PIM_OK) {
                std::cout << "Abort" << std::endl;
                return;
            }

            pimFree(output_obj);
            pimFree(temp_obj);

          }
        }
    }
}


// Compare PIM results with CPU reference
void compareResults(const std::vector<std::vector<std::vector<std::vector<float>>>> &pim_output,
                   const std::vector<std::vector<std::vector<std::vector<float>>>> &cpu_output)
{
  if (pim_output.size() != cpu_output.size()) {
    std::cout << "[ERROR] Output sizes don't match!" << std::endl;
    return;
  }
  
  int batch_size = pim_output.size();
  int num_features = pim_output[0].size();
  int height = pim_output[0][0].size();
  int width = pim_output[0][0][0].size();
  
  float max_diff = 0.0f;
  float total_diff = 0.0f;
  int total_elements = 0;
  
  for (int b = 0; b < batch_size; b++) {
    for (int f = 0; f < num_features; f++) {
      for (int h_idx = 0; h_idx < height; h_idx++) {
        for (int w_idx = 0; w_idx < width; w_idx++) {
          float diff = std::abs(pim_output[b][f][h_idx][w_idx] - cpu_output[b][f][h_idx][w_idx]);
          max_diff = std::max(max_diff, diff);
          total_diff += diff;
          total_elements++;
        }
      }
    }
  }
  
  float avg_diff = total_diff / total_elements;
  
  std::cout << "[INFO] Comparison Results:" << std::endl;
  std::cout << "  - Max difference: " << max_diff << std::endl;
  std::cout << "  - Average difference: " << avg_diff << std::endl;
  std::cout << "  - Total elements: " << total_elements << std::endl;
  
  if (avg_diff < 1e-2) {
    std::cout << "[SUCCESS] PIM and CPU results match within tolerance!" << std::endl;
  } else {
    std::cout << "[WARNING] PIM and CPU results have significant differences!" << std::endl;
  }
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  
  std::cout << "[INFO] Batch Normalization Parameters:" << std::endl;
  std::cout << "  - Batch size: " << params.batch_size << std::endl;
  std::cout << "  - Features: " << params.num_features << std::endl;
  std::cout << "  - Height: " << params.height << std::endl;
  std::cout << "  - Width: " << params.width << std::endl;
  std::cout << "  - Epsilon: " << params.eps << std::endl;
  std::cout << "  - Affine: " << (params.affine ? "true" : "false") << std::endl;
  std::cout << "  - Verify with CPU: " << (params.shouldVerify ? "true" : "false") << std::endl;
  
  // Generate random input tensor
  std::vector<std::vector<std::vector<std::vector<float>>>> input_tensor;
  generateRandomInput(params.batch_size, params.num_features, params.height, params.width, input_tensor);
  
  // Initialize PIM device after generating input data
  if (!createDevice(params.dramConfigFile)) {
    std::cout << "Abort: createDevice failed" << std::endl;
    return -1;
  }
  
  // PIM computation
  std::vector<std::vector<std::vector<std::vector<float>>>> pim_output;
  pimBatchNorm(input_tensor, pim_output, params.num_features, params.eps, params.affine);
  pimShowStats();
  
  // CPU reference computation only if verification is enabled
  if (params.shouldVerify) {
    std::vector<std::vector<std::vector<std::vector<float>>>> cpu_output;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpuBatchNorm(input_tensor, cpu_output, params.num_features, params.eps, params.affine);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    
    std::cout << "[INFO] CPU computation time: " << cpu_duration.count() / 1000.0 << " ms" << std::endl;
    
    // Compare results
    compareResults(pim_output, cpu_output);
  }
  
  
  return 0;
} 