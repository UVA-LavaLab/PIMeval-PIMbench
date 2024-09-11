/**
 * @file vec-add.cpp
 * @brief Vector Addition.
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <unistd.h>
#include <getopt.h>
#include <chrono>
#include <omp.h>
#include "../../../utilBaselines.h"

// Global Vectors
vector<int32_t> A;
vector<int32_t> B;
vector<int32_t> C;

/**
 * @brief CPU vector addition kernel
 * @param numElements Number of elements in the vectors
 */
static void vectorAddition(uint64_t numElements)
{
#pragma omp parallel for
    for (uint64_t i = 0; i < numElements; i++)
    {
        C[i] = A[i] + B[i];
    }
}

// Struct for Parameters
struct Params
{
    uint64_t vectorSize = 1024; // Default vector size
};

/**
 * @brief Displays usage information
 */
void usage()
{
    cerr << "\nUsage:  ./vec-add.out [options]\n"
         << "\nOptions:\n"
         << "    -l    vector size (default=1024 elements)\n"
         << "    -h    display this help message\n";
}

/**
 * @brief Parses command line input parameters
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return Parsed parameters
 */
Params parseParams(int argc, char **argv)
{
    Params params;

    int opt;
    while ((opt = getopt(argc, argv, "l:h")) != -1)
    {
        switch (opt)
        {
        case 'h':
            usage();
            exit(0);
        case 'l':
            params.vectorSize = stoull(optarg);
            break;
        default:
            cerr << "\nUnrecognized option: " << opt << "\n";
            usage();
            exit(1);
        }
    }

    return params;
}

/**
 * @brief Main function.
 */
int main(int argc, char **argv)
{
    // Parse input parameters
    Params params = parseParams(argc, argv);
    uint64_t vectorSize = params.vectorSize;
    std::cout << "Running vector addition for CPU on vector of size: " << vectorSize << std::endl;

    C.resize(vectorSize);

    // Initialize vectors
    getVector<int32_t>(vectorSize, A);
    getVector<int32_t>(vectorSize, B);
    std::cout << "Vector Initialization done!" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < WARMUP; ++i)
    {
        vectorAddition(vectorSize);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, milli> elapsedTime = (end - start) / WARMUP;
    std::cout << "Finished Running.\nDuration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << std::endl;

    return 0;
}
