#include <cmath>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <unistd.h>
#include <iomanip>

#include "utilBaselines.h"

#define BLOCK_SIZE 256

using namespace std;

struct Params {
    uint64_t dataSize = 2048;
    int epochs = 1000;
    float learningRate = 0.01f;
    string inputFile = "";
};

void usage() {
    fprintf(stderr,
        "\nUsage:  ./lr_gpu.out [options]"
        "\n"
        "\n    -l    input size (default=2048 elements)"
        "\n    -e    number of epochs (default=1000)"
        "\n    -r    learning rate (default=0.01)"
        "\n    -i    input file (not implemented)"
        "\n");
}

Params getInputParams(int argc, char** argv) {
    Params p;
    int opt;
    while ((opt = getopt(argc, argv, "h:l:e:r:i:")) >= 0) {
        switch (opt) {
            case 'h': usage(); exit(0);
            case 'l': p.dataSize = strtoull(optarg, nullptr, 0); break;
            case 'e': p.epochs = atoi(optarg); break;
            case 'r': p.learningRate = atof(optarg); break;
            case 'i': p.inputFile = optarg; break;
            default: fprintf(stderr, "\nUnrecognized option!\n"); usage(); exit(0);
        }
    }
    return p;
}

__device__ float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void computeGradientsUpdate(float* w, float* b, const int* X, const int* Y, int n, float lr) {
    __shared__ float dw_shared[BLOCK_SIZE];
    __shared__ float db_shared[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float dw_local = 0.0f, db_local = 0.0f;
    if (idx < n) {
        float z = (*w) * X[idx] + (*b);
        float pred = sigmoid(z);
        float error = pred - Y[idx];
        dw_local = error * X[idx];
        db_local = error;
    }

    dw_shared[tid] = dw_local;
    db_shared[tid] = db_local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            dw_shared[tid] += dw_shared[tid + stride];
            db_shared[tid] += db_shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile float* v_dw = dw_shared;
        volatile float* v_db = db_shared;
        warpReduce(v_dw, tid);
        warpReduce(v_db, tid);
    }

    if (tid == 0) {
        atomicAdd(w, -lr * dw_shared[0] / n);
        atomicAdd(b, -lr * db_shared[0] / n);
    }
}

void getVector(uint64_t size, vector<int>& vec) {
    vec.resize(size);
    for (uint64_t i = 0; i < size; ++i)
        vec[i] = rand() % 16;
}



int main(int argc, char* argv[]) {
    Params params = getInputParams(argc, argv);
    uint64_t n = params.dataSize;

    vector<int> X, Y;
    getVector(n, X);
    getVector(n, Y);
    for (auto& y : Y) y = y % 2;

    float w_gpu = 0.0f, b_gpu = 0.0f;
    int* d_X, * d_Y;
    float *d_w, *d_b;
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMalloc(&d_X, n * sizeof(int));
    cudaMalloc(&d_Y, n * sizeof(int));
    cudaMalloc(&d_w, sizeof(float));
    cudaMalloc(&d_b, sizeof(float));

    cudaMemcpy(d_X, X.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, &w_gpu, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b_gpu, sizeof(float), cudaMemcpyHostToDevice);

    double gpu_time_ms;
    auto [gpuElapsed, _, __] = measureCUDAPowerAndElapsedTime([&]() {
        for (int epoch = 0; epoch < params.epochs; ++epoch) {
            computeGradientsUpdate<<<numBlocks, BLOCK_SIZE>>>(d_w, d_b, d_X, d_Y, n, params.learningRate);
        }
        cudaDeviceSynchronize();
    });
    gpu_time_ms = gpuElapsed;

    cudaMemcpy(&w_gpu, d_w, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b_gpu, d_b, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_w);
    cudaFree(d_b);

    cout << "[GPU] Duration: " << gpu_time_ms << " ms\n";
    cout << "[GPU] Model: sigmoid(" << w_gpu << " * x + " << b_gpu << ")\n";

    double w_cpu = 0.0, b_cpu = 0.0;
    auto start = chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < params.epochs; ++epoch) {
        double dw = 0.0, db = 0.0;
        // #pragma omp parallel for reduction(+ : dw, db)
        for (uint64_t i = 0; i < n; i++) {
            double z = w_cpu * X[i] + b_cpu;
            double pred = 1.0 / (1.0 + exp(-z));
            double error = pred - Y[i];
            dw += error * X[i];
            db += error;
        }
        w_cpu -= params.learningRate * dw / n;
        b_cpu -= params.learningRate * db / n;
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;
    cout << "[CPU] Duration: " << fixed << setprecision(3) << elapsed.count() << " ms\n";
    cout << "[CPU] Model: sigmoid(" << w_cpu << " * x + " << b_cpu << ")\n";

    if (abs(w_cpu - w_gpu) > 1e-4 || abs(b_cpu - b_gpu) > 1e-4) {
        cout << "[CHECK] Mismatch between CPU and GPU results!\n";
    } else {
        cout << "[CHECK] CPU and GPU results match.\n";
    }

    return 0;
}
