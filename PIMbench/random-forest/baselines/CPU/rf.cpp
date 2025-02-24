/**
 * @file knn.cpp
 * @brief Template for a Host Application Source File.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <unordered_map>
#include <chrono>
#include <vector>
#include <string>
#include <queue>
#include <iomanip>
using namespace std;

#include <omp.h>
#include "../../../utilBaselines.h"
#include <cfloat>


vector<vector<int>> negativeParameterMapping;
vector<vector<int>> modelParameter;
vector<vector<int>> compareParameterMapping;

// Params ---------------------------------------------------------------------
typedef struct Params
{
    int tree_count;
    int dimension;
    int numThreads;
    int treeDepth;
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
            "\n");
}

struct Params input_params(int argc, char **argv)
{
    struct Params p;
    p.dimension = 10;
    p.tree_count = 100;
    p.numThreads = 8;
    p.treeDepth = 5;


    int opt;
    while ((opt = getopt(argc, argv, "h:t:n:d:")) >= 0)
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
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
        }
    }

    return p;
}

// inline int calculateDistance(const vector<int> &pointA, const vector<int> &pointB, int dim) {
//     int sum = 0;
//     for (int i = 0; i < dim; i++)
//     {
//         sum += abs(pointA[i] - pointB[i]);
//     }
//     return sum;
// }

// struct DistancePoint {
//     double distance;
//     int index;
    
//     DistancePoint(double d, int i) : distance(d), index(i) {}
// };

// struct CompareDistance {
//     bool operator()(const DistancePoint& dp1, const DistancePoint& dp2) {
//         return dp1.distance > dp2.distance;
//     }
// };

// void runKNN(uint64_t numPoints, uint64_t numTests, int k, int dim, int numThreads, int target, vector<int> testPredictions)
// {
//     vector<priority_queue<DistancePoint, vector<DistancePoint>, CompareDistance>> localMinHeaps(numTests);
//     omp_set_num_threads(numThreads);

// #pragma omp parallel for
//     for (uint64_t i = 0; i < numTests; ++i)
//     {
//         for (uint64_t j = 0; j < numPoints; ++j) {
//             double dist = calculateDistance(dataPoints[i], dataPoints[j], dim);
//             if (int(localMinHeaps[i].size()) < k) {
//                 localMinHeaps[i].emplace(dist, j);
//             } else if (dist < localMinHeaps[i].top().distance) {
//                 localMinHeaps[i].pop();
//                 localMinHeaps[i].emplace(dist, j);
//             }

//         }
//     }
//     for (uint64_t i = 0; i < numTests; ++i) {
//         // Tally the labels of the k nearest neighbors
//         unordered_map<int, int> labelCount;
//         while (!localMinHeaps[i].empty()) {
//             int index = localMinHeaps[i].top().index;
//             int label = dataPoints[index][target];
//             labelCount[label]++;
//             localMinHeaps[i].pop();
//         }
        
//         // Find the label with the highest count
//         int maxCount = 0;
//         int bestLabel = -1;
//         for (const auto& entry : labelCount) {
//             if (entry.second > maxCount) {
//                 maxCount = entry.second;
//                 bestLabel = entry.first;
//             }
//         }

//         // Assign the most frequent label to the test point
//         testPredictions[i] = bestLabel;
//     }

    
// }

void getModel(uint64_t numRows, uint64_t numCols, vector<vector<int>> &matrix, vector<vector<int>> &mappingMatrix, vector<vector<int>> &compareMatrix)
{
    // Seed the random number generator with a fixed seed for reproducibility
    srand(8746219);

    // Resize the matrix to the specified number of rows and columns
    matrix.resize(numRows, vector<int>(numCols));
    mappingMatrix.resize(numRows, vector<int>(numCols));
    compareMatrix.resize(numRows, vector<int>(numCols));

    #pragma omp parallel for
    for (uint64_t row = 0; row < numRows; ++row) {
        for (uint64_t col = 0; col < numCols; ++col) {

            matrix[row][col] = (rand() % 2001) - 1000;  // Generates a number in [-1000, 1000]
            
            // to simulate compare operations at each leaf 
            int rand_op = rand() % 4;
            compareMatrix[row][col] = rand_op;

            // if its > or >=, switch operators and negate parameter
            if (rand_op == 2 or rand_op == 3) {
                mappingMatrix[row][col] = -1;  // indicates that input needs to be negated
                matrix[row][col] = -1 * matrix[row][col];  // negate mdoel parameter
            } else mappingMatrix[row][col] = 1;
        }
    }
}

void runRF() {


}


int main(int argc, char **argv)
{
    struct Params params = input_params(argc, argv);
    int numberOfPaths = ((int) pow(2, params.treeDepth)) * params.tree_count;

    std::unordered_map<int, std::string> compMap;
    compMap[0] = "<=";
    compMap[1] = "<";
    compMap[2] = ">=";
    compMap[3] = ">";


    // using random matrix to simulate training an RF classifer 
    getModel(numberOfPaths, params.dimension, modelParameter, negativeParameterMapping, compareParameterMapping);
    std::cout << modelParameter[0][0] << endl;


    // generate random inputs
    vector<int> modelInput(params.dimension);
    //mappingMatrix.resize(param);
    for(int i = 0; i < params.dimension; i++) {
        modelInput[i] = (rand() % 501) - 500;  // generate random inputs from -500 to 500
    }

    // negate input if model param was negated


    // NOTE - 2/23/25: put CPU implementation on hold until its determined how it should be implemented: on pytroch or mimicking PIM 
    


    // uint64_t numPoints = params.numDataPoints, numTests = params.numTestPoints;
    // int k = params.k, dim = params.dimension;
    // int target = params.target;

    std::cout << "Set up done!\n";
    
    auto start = chrono::high_resolution_clock::now();
    //runKNN(numPoints, numTests, k, dim, params.numThreads, target, testPredictions);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> elapsedTime = (end - start);
    std::cout << "Duration: " << fixed << setprecision(3) << elapsedTime.count() << " ms." << endl;
    return 0;
}
