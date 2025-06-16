#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <iomanip>
#include <vector>

#include "utilBaselines.h"

using namespace std;

struct Params {
    uint64_t dataSize = 2048;
    int epochs = 1000;
    float learningRate = 0.01f;
    string inputFile = "";
};

void usage() {
    fprintf(stderr,
        "\nUsage:  ./lr_host.out [options]"
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
            case 'i': 
                p.inputFile = optarg; 
                break;
            default: 
                fprintf(stderr, "\nUnrecognized option!\n"); 
                usage(); 
                exit(0);
        }
    }
    return p;
}

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

int main(int argc, char* argv[]) {
    Params params = getInputParams(argc, argv);

    if (!params.inputFile.empty()) {
        std::cout << "Reading input from file is not yet implemented." << std::endl;
        return 1;
    }

    uint64_t n = params.dataSize;
    vector<int32_t> dataPointsX, dataPointsY;
    getVector(n, dataPointsX);
    getVector(n, dataPointsY);

    for (uint64_t i = 0; i < n; i++) {
        dataPointsY[i] = dataPointsY[i] % 2;
    }

    cout << "Done initializing data\n";

    float w = 0.0, b = 0.0;
    auto start = chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < params.epochs; ++epoch) {
        float dw = 0.0, db = 0.0;

        #pragma omp parallel for reduction(+ : dw, db)
        for (uint64_t i = 0; i < n; i++) {
            float z = w * dataPointsX[i] + b;
            float pred = sigmoid(z);
            float error = pred - dataPointsY[i];

            dw += error * dataPointsX[i];
            db += error;
        }

        w -= params.learningRate * dw / n;
        b -= params.learningRate * db / n;
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsedTime = end - start;

    cout << "Duration: " << fixed << setprecision(3) << elapsedTime.count() << " ms\n";
    cout << "Model: sigmoid(" << w << " * x + " << b << ")\n";

    return 0;
}
