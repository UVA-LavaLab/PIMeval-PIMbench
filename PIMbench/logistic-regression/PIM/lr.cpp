#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <cmath>
#include <chrono>
#include <iomanip>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "util.h"
#include "libpimeval.h"

using namespace std;

struct Params {
    uint64_t dataSize = 2048;
    int epochs = 1000;
    float learningRate = 0.01f;
    char* configFile = nullptr;
    char* inputFile = nullptr;
    bool shouldVerify = false;
};

void usage() {
    fprintf(stderr,
        "\nUsage:  ./lr.out [options]"
        "\n"
        "\n    -l    input size (default=2048 elements)"
        "\n    -e    number of epochs (default=1000)"
        "\n    -r    learning rate (default=0.01)"
        "\n    -c    DRAMsim config file"
        "\n    -i    input file (not implemented)"
        "\n    -v    t = verify with host output"
        "\n");
}

Params getInputParams(int argc, char** argv) {
    Params p;
    int opt;
    while ((opt = getopt(argc, argv, "h:l:e:r:c:i:v:")) >= 0) {
        switch (opt) {
            case 'h': 
                usage(); 
                exit(0);
            case 'l': 
                p.dataSize = strtoull(optarg, nullptr, 0); 
                break;
            case 'e': 
                p.epochs = atoi(optarg); 
                break;
            case 'r': 
                p.learningRate = atof(optarg); 
                break;
            case 'c': 
                p.configFile = optarg; 
                break;
            case 'i': 
                p.inputFile = optarg; 
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

float sigmoid_exact(float z) {
    return 1.0f / (1.0f + exp(-z));
}


std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();

void runLogisticRegressionPIM(uint64_t dataSize, int epochs, float lr, const vector<int>& X, const vector<int>& Y, float& w, float& b) {
    PimObjId xObj = pimAlloc(PIM_ALLOC_AUTO, dataSize, PIM_FP32);
    if (xObj == -1)
    {
        std::cout << "Abort" << std::endl;
        return;
    }
    PimObjId yObj = pimAllocAssociated(xObj, PIM_FP32);
    if (yObj == -1)
    {
        std::cout << "Abort" << std::endl;
        return;
    }
    PimObjId predictionObj = pimAllocAssociated(xObj, PIM_FP32);
    if (predictionObj == -1)
    {
        std::cout << "Abort" << std::endl;
        return;
    }
    PimObjId errorObj = pimAllocAssociated(xObj, PIM_FP32);
    if (errorObj == -1)
    {
        std::cout << "Abort" << std::endl;
        return;
    }


    auto start_cpu = std::chrono::high_resolution_clock::now();

    std::vector<float> Xf(dataSize), Yf(dataSize);
    for (uint64_t i = 0; i < dataSize; ++i) {
        Xf[i] = static_cast<float>(X[i]);
        Yf[i] = static_cast<float>(Y[i]);
    }

    auto stop_cpu = std::chrono::high_resolution_clock::now();
    // hostElapsedTime += (stop_cpu - start_cpu);


    PimStatus status = pimCopyHostToDevice(Xf.data(), xObj);
    if (status != PIM_OK)
    {
        std::cout << "Abort" << std::endl;
        return;
    }

    status = pimCopyHostToDevice(Yf.data(), yObj);
    if (status != PIM_OK)
    {
        std::cout << "Abort" << std::endl;
        return;
    }
    

    std::vector<float> zBuffer(dataSize);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float dw = 0.0f, db = 0.0f;

        status = pimMulScalar(xObj, predictionObj, *(uint64_t*)&w);
        if (status != PIM_OK)
        {
          std::cout << "Abort" << std::endl;
          return;
        }

        status = pimAddScalar(predictionObj, predictionObj, *(uint64_t*)&b);
        if (status != PIM_OK)
        {
          std::cout << "Abort" << std::endl;
          return;
        }
        status = pimCopyDeviceToHost(predictionObj, zBuffer.data());
        if (status != PIM_OK)
        {
          std::cout << "Abort" << std::endl;
          return;
        }


        auto start_cpu = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (uint64_t i = 0; i < dataSize; ++i) {
            zBuffer[i] = sigmoid_exact(zBuffer[i]);
        }
        auto stop_cpu = std::chrono::high_resolution_clock::now();
        hostElapsedTime += (stop_cpu - start_cpu);

        status = pimCopyHostToDevice(zBuffer.data(), predictionObj);
        if (status != PIM_OK)
        {
          std::cout << "Abort" << std::endl;
          return;
        }

        status = pimSub(predictionObj, yObj, errorObj);
        if (status != PIM_OK)
        {
          std::cout << "Abort" << std::endl;
          return;
        }

        status = pimMul(errorObj, xObj, predictionObj);
        if (status != PIM_OK)
        {
          std::cout << "Abort" << std::endl;
          return;
        }

        status = pimRedSum(predictionObj, &dw);
        if (status != PIM_OK)
        {
          std::cout << "Abort" << std::endl;
          return;
        }
        
        status = pimRedSum(errorObj, &db);
        if (status != PIM_OK)
        {
          std::cout << "Abort" << std::endl;
          return;
        }

        w -= lr * dw / dataSize;
        b -= lr * db / dataSize;
    }

    pimFree(xObj);
    pimFree(yObj);
    pimFree(predictionObj);
    pimFree(errorObj);
    std::cout << "Host elapsed time: " << hostElapsedTime.count() << " ms" << std::endl;
}




void runLogisticRegressionHost(uint64_t n, int epochs, float lr, const vector<int>& X, const vector<int>& Y, float& w_host, float& b_host) {

    w_host = 0.0f;
    b_host = 0.0f;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float dw = 0.0f, db = 0.0f;
        for (uint64_t i = 0; i < n; ++i) {
            float z = w_host * X[i] + b_host;
            float pred = sigmoid_exact(z);
            float error = pred - Y[i];
            dw += error * X[i];
            db += error;
        }
        w_host -= lr * dw / n;
        b_host -= lr * db / n;
    }
}


int main(int argc, char* argv[]) {
    Params params = getInputParams(argc, argv);
    vector<int> X(params.dataSize), Y(params.dataSize);
    
    if (params.inputFile == nullptr){
        getVector(params.dataSize, X);
        getVector(params.dataSize, Y);
        for (auto& y : Y) y = y % 2;
    }
    else{
        std::cout << "Reading from input file is not implemented yet." << std::endl;
        return 1;
    }
    
    if (!createDevice(params.configFile)) return 1;


    float w = 0.0f, b = 0.0f;


    runLogisticRegressionPIM(params.dataSize, params.epochs, params.learningRate, X, Y, w, b);
    pimShowStats();

    cout << "Model: sigmoid(" << w << " * x + " << b << ")\n";


    if (params.shouldVerify) {
        float w_host, b_host;
        runLogisticRegressionHost(params.dataSize, params.epochs, params.learningRate, X, Y, w_host, b_host);
    
        cout << "Host Model: sigmoid(" << w_host << " * x + " << b_host << ")\n";
    
        float w_diff = fabs(w - w_host);
        float b_diff = fabs(b - b_host);
    
        if (w_diff < 1e-4 && b_diff < 1e-4)
            cout << "Verification PASSED.\n";
        else
            cout << "Verification FAILED.\n";
    }

    return 0;
}
