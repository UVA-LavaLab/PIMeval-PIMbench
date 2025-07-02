// Layer Normalization on CPU and PIM
// Copyright (c) 2024 University of Virginia
// Licensed under the MIT License

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

struct Params
{
  uint64_t batch_size, num_features, height, width;
  float eps;
  bool affine;
  char *dramConfigFile;
  bool shouldVerify;
};

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./layer_norm.out [options]"
          "\n"
          "\n    -b    batch size (default=64)"
          "\n    -f    number of features (default=64)"
          "\n    -r    height (default=224)"
          "\n    -c    width (default=224)"
          "\n    -e    epsilon value (default=1e-5)"
          "\n    -a    affine transformation (default=false)"
          "\n    -d    DRAM config file"
          "\n    -v    should verify result with CPU"
          "\n");
}

Params getInputParams(int argc, char **argv)
{
  Params p = {64, 64, 224, 224, 1e-5, false, nullptr, false};
  int opt;
  while ((opt = getopt(argc, argv, "hb:f:r:c:e:a:d:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
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
      p.affine = (*optarg == 't');
      break;
    case 'd':
      p.dramConfigFile = optarg;
      break;
    case 'v':
      p.shouldVerify = (*optarg == 't');
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

void generateRandomInput(int B, int C, int H, int W, vector<vector<vector<vector<float>>>> &tensor)
{
  std::mt19937 gen(std::random_device{}());
  std::normal_distribution<float> dist(0.0, 1.0);
  tensor.resize(B, vector<vector<vector<float>>>(C, vector<vector<float>>(H, vector<float>(W))));
  for (int b = 0; b < B; ++b)
    for (int c = 0; c < C; ++c)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
          tensor[b][c][h][w] = dist(gen);
}

void compareResults(const vector<vector<vector<vector<float>>>> &pim_out,
                    const vector<vector<vector<vector<float>>>> &cpu_out)
{
  float max_diff = 0.0f;
  float total_diff = 0.0f;
  int total_elements = 0;

  for (size_t b = 0; b < pim_out.size(); ++b)
    for (size_t c = 0; c < pim_out[0].size(); ++c)
      for (size_t h = 0; h < pim_out[0][0].size(); ++h)
        for (size_t w = 0; w < pim_out[0][0][0].size(); ++w)
        {
          float diff = fabs(pim_out[b][c][h][w] - cpu_out[b][c][h][w]);
          max_diff = max(max_diff, diff);
          total_diff += diff;
          total_elements++;
        }

  float avg_diff = total_diff / total_elements;

  cout << "[INFO] Comparison Results:\n";
  cout << "  - Max difference: " << max_diff << endl;
  cout << "  - Average difference: " << avg_diff << endl;
  cout << "  - Total elements: " << total_elements << endl;

  if (avg_diff < 1e-3)
    cout << "[SUCCESS] PIM and CPU outputs match within tolerance." << endl;
  else
    cout << "[WARNING] PIM and CPU outputs diverge beyond tolerance!" << endl;
}

void cpuLayerNorm(const vector<vector<vector<vector<float>>>> &input,
                  vector<vector<vector<vector<float>>>> &output,
                  float eps, bool affine)
{
  int B = input.size();
  int C = input[0].size();
  int H = input[0][0].size();
  int W = input[0][0][0].size();

  output.resize(B, vector<vector<vector<float>>>(C, vector<vector<float>>(H, vector<float>(W))));
  vector<float> gamma(C, 1.0f), beta(C, 0.0f);
  if (affine)
  {
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(1.0, 0.1);
    for (int i = 0; i < C; ++i)
    {
      gamma[i] = dist(gen);
      beta[i] = dist(gen);
    }
  }

  for (int b = 0; b < B; ++b)
  {
    float sum = 0.0f, sum_sq = 0.0f;
    int count = C * H * W;
    for (int c = 0; c < C; ++c)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
        {
          float val = input[b][c][h][w];
          sum += val;
          sum_sq += val * val;
        }

    float mean = sum / count;
    float var = max((sum_sq / count) - (mean * mean), 0.0f);
    float std_dev = sqrt(var + eps);

    for (int c = 0; c < C; ++c)
      for (int h = 0; h < H; ++h)
        for (int w = 0; w < W; ++w)
        {
          float normed = (input[b][c][h][w] - mean) / std_dev;
          output[b][c][h][w] = affine ? gamma[c] * normed + beta[c] : normed;
        }
  }
}

void pimLayerNorm(const std::vector<std::vector<std::vector<std::vector<float>>>> &input,
                  std::vector<std::vector<std::vector<std::vector<float>>>> &output,
                  float eps, bool affine)
{
  int B = input.size();
  int C = input[0].size();
  int H = input[0][0].size();
  int W = input[0][0][0].size();
  int count = C * H * W;

  // Prepare output tensor
  output.resize(B);
  for (int b = 0; b < B; b++)
  {
    output[b].resize(C);
    for (int f = 0; f < C; f++)
    {
      output[b][f].resize(H);
      for (int h = 0; h < H; h++)
        output[b][f][h].resize(W);
    }
  }

  // Affine params
  std::vector<float> gamma(C, 1.0f), beta(C, 0.0f);
  if (affine)
  {
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(1.0f, 0.1f);
    for (int f = 0; f < C; ++f)
    {
      gamma[f] = dist(gen);
      beta[f] = dist(gen);
    }
  }

  PimObjId row_obj = pimAlloc(PIM_ALLOC_AUTO, W, PIM_FP32);
  if (row_obj == -1)
  {
    std::cerr << "Abort: pimAlloc failed for row_obj\n";
    return;
  }

  PimObjId mean_obj = pimAllocAssociated(row_obj, PIM_FP32);
  if (mean_obj == -1)
  {
    std::cerr << "Abort: pimAlloc failed for mean_obj\n";
    return;
  }

  PimObjId std_obj = pimAllocAssociated(row_obj, PIM_FP32);
  if (std_obj == -1)
  {
    std::cerr << "Abort: pimAlloc failed for std_obj\n";
    return;
  }

  PimObjId affine_std = pimAllocAssociated(row_obj, PIM_FP32);
  if (affine_std == -1)
  {
    std::cerr << "Abort: pimAlloc failed for affine_std\n";
    return;
  }

  PimObjId affine_mean = pimAllocAssociated(row_obj, PIM_FP32);
  if (affine_mean == -1)
  {
    std::cerr << "Abort: pimAlloc failed for affine_mean\n";
    return;
  }

  for (int b = 0; b < B; ++b)
  {
    float sum = 0.0f, sum_sq = 0.0f;

    // Compute sum and sum_sq over all rows [f][h]
    for (int f = 0; f < C; ++f)
    {
      for (int h = 0; h < H; ++h)
      {
        PimStatus status = pimCopyHostToDevice((void *)input[b][f][h].data(), row_obj);
        if (status != PIM_OK)
        {
          std::cerr << "Abort: copy row failed\n";
          return;
        }

        float row_sum = 0.0f;
        if (pimRedSum(row_obj, &row_sum) != PIM_OK)
        {
          std::cerr << "Abort: pimRedSum failed\n";
          return;
        }
        sum += row_sum;

        if (pimMul(row_obj, row_obj, row_obj) != PIM_OK)
        {
          std::cerr << "Abort: pimMul failed\n";
          return;
        }

        float row_sq_sum = 0.0f;
        if (pimRedSum(row_obj, &row_sq_sum) != PIM_OK)
        {
          std::cerr << "Abort: pimRedSum square failed\n";
          return;
        }
        sum_sq += row_sq_sum;
      }
    }

    float mean = sum / count;
    float var = (sum_sq / count) - (mean * mean);
    float std_dev = std::sqrt(std::max(var, 0.0f) + eps);
    std::cout << "Sample " << b << " Mean: " << mean << ", Std Dev: " << std_dev << std::endl;

    if (pimBroadcastFP(mean_obj, mean) != PIM_OK)
    {
      std::cerr << "Abort: mean broadcast failed\n";
      return;
    }

    if (pimBroadcastFP(std_obj, std_dev) != PIM_OK)
    {
      std::cerr << "Abort: std broadcast failed\n";
      return;
    }

    // Normalize each row [f][h]
    for (int f = 0; f < C; ++f)
    {
      for (int h = 0; h < H; ++h)
      {
        PimStatus status = pimCopyHostToDevice((void *)input[b][f][h].data(), row_obj);
        if (status != PIM_OK)
        {
          std::cerr << "Abort: copy row failed\n";
          return;
        }

        // Normalize
        if (pimSub(row_obj, mean_obj, row_obj) != PIM_OK)
        {
          std::cerr << "Abort: mean subtraction failed\n";
          return;
        }

        if (pimDiv(row_obj, std_obj, row_obj) != PIM_OK)
        {
          std::cerr << "Abort: std division failed\n";
          return;
        }
        // // Affine
        if (affine)
        {
          status = pimBroadcastFP(affine_std, gamma[f]);
          if (status != PIM_OK)
          {
            std::cout << "Abort" << std::endl;
            return;
          }

          status = pimMul(row_obj, affine_std, row_obj);
          if (status != PIM_OK)
          {
            std::cout << "Abort" << std::endl;
            return;
          }

          status = pimBroadcastFP(affine_mean, beta[f]);
          if (status != PIM_OK)
          {
            std::cout << "Abort" << std::endl;
            return;
          }

          status = pimAdd(row_obj, affine_mean, row_obj);
          if (status != PIM_OK)
          {
            std::cout << "Abort" << std::endl;
            return;
          }
        }

        // Copy back
        if (pimCopyDeviceToHost(row_obj, output[b][f][h].data()) != PIM_OK)
        {
          std::cerr << "Abort: copy back failed\n";
          return;
        }
      }
    }
  }
  pimFree(row_obj);
  pimFree(std_obj);
  pimFree(mean_obj);
}

int main(int argc, char *argv[])
{
  Params params = getInputParams(argc, argv);
  cout << "[INFO] Running LayerNorm with Batch=" << params.batch_size
       << ", Features=" << params.num_features
       << ", Height=" << params.height
       << ", Width=" << params.width << endl;
  vector<vector<vector<vector<float>>>> input, cpu_out, pim_out;
  generateRandomInput(params.batch_size, params.num_features, params.height, params.width, input);
  if (!createDevice(params.dramConfigFile))
  {
    cerr << "[ERROR] createDevice failed" << endl;
    return -1;
  }
  pimLayerNorm(input, pim_out, params.eps, params.affine);
  pimShowStats();
  if (params.shouldVerify)
  {
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpuLayerNorm(input, cpu_out, params.eps, params.affine);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    compareResults(pim_out, cpu_out);
    cout << "[INFO] CPU LayerNorm time: " << cpu_duration.count() << " ms" << endl;
  }
  return 0;
}
