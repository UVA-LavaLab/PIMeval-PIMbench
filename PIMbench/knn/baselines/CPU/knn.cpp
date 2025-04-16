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
#include "utilBaselines.h"
#include <cfloat>

vector<vector<int>> dataPoints;
vector<vector<int>> testPoints;

// Params ---------------------------------------------------------------------
typedef struct Params
{
    uint64_t numTestPoints;
    uint64_t numDataPoints;
    int dimension;
    int k;
    int numThreads;
    char *inputTestFile;
    char *inputDataFile;
    int target;
} Params;

void usage()
{
    fprintf(stderr,
            "\nUsage:  ./knn.out [options]"
            "\n"
            "\n    -t    # of threads (default=8)"
            "\n    -n    number of reference points (default=65536 points)"
            "\n    -m    number of query points (default=100 points)"
            "\n    -d    dimension (default=2)"
            "\n    -k    value of K (default=20)"
            "\n    -x    target dimension index of the data set(default=1)"
            "\n    -i    input file containing training datapoints (default=generates datapoints with random numbers)"
            "\n    -j    input file containing testing datapoints (default=generates datapoints with random numbers)"
            "\n");
}

struct Params input_params(int argc, char **argv)
{
    struct Params p;
    p.numDataPoints = 65536;
    p.numTestPoints = 100;
    p.dimension = 2;
    p.k = 20;
    p.numThreads = 8;
    p.target = 1;
    p.inputTestFile = nullptr;
    p.inputDataFile = nullptr;

    int opt;
    while ((opt = getopt(argc, argv, "h:k:t:n:m:d:x:i:j:")) >= 0)
    {
        switch (opt)
        {
        case 'h':
            usage();
            exit(0);
            break;
        case 'n':
            p.numDataPoints = atoll(optarg);
            break;
        case 'm':
            p.numTestPoints = atoll(optarg);
            break;
        case 'd':
            p.dimension = atoll(optarg);
            p.target = p.dimension - 1;
            break;
        case 'k':
            p.k = atoi(optarg);
            break;
        case 't':
            p.numThreads = atoi(optarg);
            break;
        case 'x':
            p.target = atoi(optarg);
            break;
        case 'i':
            p.inputDataFile = optarg;
            break;
        case 'j':
            p.inputTestFile = optarg;
            break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
        }
    }

    return p;
}

inline int calculateDistance(const vector<int> &pointA, const vector<int> &pointB, int dim) {
    int sum = 0;
    for (int i = 0; i < dim; i++)
    {
        sum += abs(pointA[i] - pointB[i]);
    }
    return sum;
}

struct DistancePoint {
    double distance;
    int index;
    
    DistancePoint(double d, int i) : distance(d), index(i) {}
};

struct CompareDistance {
    bool operator()(const DistancePoint& dp1, const DistancePoint& dp2) {
        return dp1.distance > dp2.distance;
    }
};

void runKNN(uint64_t numPoints, uint64_t numTests, int k, int dim, int numThreads, int target, vector<int> testPredictions)
{
    vector<priority_queue<DistancePoint, vector<DistancePoint>, CompareDistance>> localMinHeaps(numTests);
    omp_set_num_threads(numThreads);

#pragma omp parallel for
    for (uint64_t i = 0; i < numTests; ++i)
    {
        for (uint64_t j = 0; j < numPoints; ++j) {
            double dist = calculateDistance(dataPoints[i], dataPoints[j], dim);
            if (int(localMinHeaps[i].size()) < k) {
                localMinHeaps[i].emplace(dist, j);
            } else if (dist < localMinHeaps[i].top().distance) {
                localMinHeaps[i].pop();
                localMinHeaps[i].emplace(dist, j);
            }

        }
    }
    for (uint64_t i = 0; i < numTests; ++i) {
        // Tally the labels of the k nearest neighbors
        unordered_map<int, int> labelCount;
        while (!localMinHeaps[i].empty()) {
            int index = localMinHeaps[i].top().index;
            int label = dataPoints[index][target];
            labelCount[label]++;
            localMinHeaps[i].pop();
        }
        
        // Find the label with the highest count
        int maxCount = 0;
        int bestLabel = -1;
        for (const auto& entry : labelCount) {
            if (entry.second > maxCount) {
                maxCount = entry.second;
                bestLabel = entry.first;
            }
        }

        // Assign the most frequent label to the test point
        testPredictions[i] = bestLabel;
    }

    
}

vector<vector<int>> readCSV(const string& filename) {
    vector<vector<int>> data;
    ifstream file(filename);
    string line;
    
    if (!file.is_open()) {
        throw runtime_error("Could not open file");
    }


    while (getline(file, line)) {
        vector<int> row;
        stringstream ss(line);
        string value;
        
        while (getline(ss, value, ',')) {
            try {
                int intValue = stoi(value);
                row.push_back(intValue);
            } catch (const invalid_argument& e) {
                cerr << "Invalid argument: " << e.what() << " for value " << value << '\n';
            } catch (const out_of_range& e) {
                cerr << "Out of range: " << e.what() << " for value " << value << '\n';
            }
        }
        data.push_back(row);
    }

    file.close();
    return data;
}

/**
 * @brief Main of the Host Application.
 */
int main(int argc, char **argv)
{
    struct Params params = input_params(argc, argv);


    if (params.inputTestFile == nullptr)
    {
        getMatrix<int>(params.numTestPoints, params.dimension, testPoints);
    }
    else
    {
        vector<vector<int>> test_data_int = readCSV(params.inputTestFile);
        params.dimension = test_data_int[0].size();
        params.numTestPoints = test_data_int.size();

        testPoints = vector<vector<int>>(test_data_int.begin(), test_data_int.end());;
    }
    if (params.inputDataFile == nullptr)
    {
        getMatrix<int>(params.numDataPoints, params.dimension, dataPoints);
    }
    else
    {
        vector<vector<int>> train_data_int = readCSV(params.inputDataFile);

        params.dimension = train_data_int[0].size();
        params.numDataPoints = train_data_int.size();

        dataPoints = vector<vector<int>>(train_data_int.begin(), train_data_int.end());
    }
    uint64_t numPoints = params.numDataPoints, numTests = params.numTestPoints;
    int k = params.k, dim = params.dimension;
    int target = params.target;
    vector<int> testPredictions(numTests);

    std::cout << "Set up done!\n";
    
    auto start = chrono::high_resolution_clock::now();
    runKNN(numPoints, numTests, k, dim, params.numThreads, target, testPredictions);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> elapsedTime = (end - start);
    std::cout << "Duration: " << fixed << setprecision(3) << elapsedTime.count() << " ms." << endl;
    return 0;
}
