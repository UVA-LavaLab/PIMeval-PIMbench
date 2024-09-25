/**
 * @file radix-sort.cpp
 * @brief Template for a Host Application Source File.
 *
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <climits>
#include <omp.h>

#include "../../../utilBaselines.h"
#include <iomanip>

using namespace std;

void initVector(uint64_t vectorSize, vector<int32_t>& vectorPoints)
{
    vectorPoints.resize(vectorSize);
    // Using a fixed seed instead of time for reproducibility
    //srand((unsigned)time(NULL));
    srand(8746219);
    for (uint64_t i = 0; i < vectorSize; i++)
    {
        vectorPoints[i] = rand() % INT32_MAX;
    }
}

// Function to perform counting sort on the array based on the digit represented by exp
void countingSort(std::vector<int32_t> &dataArray, int exp, std::vector<int32_t> &output, std::vector<int32_t> &count)
{
    uint64_t n = dataArray.size();
    int numThreads = omp_get_max_threads();
    std::vector<std::vector<int>> localCount(numThreads, std::vector<int>(10, 0));

// Store count of occurrences in localCount[]
#pragma omp parallel
    {
        int threadNum = omp_get_thread_num();
#pragma omp for nowait
        for (uint64_t i = 0; i < n; i++)
        {
            localCount[threadNum][(dataArray[i] / exp) % 10]++;
        }
    }

    // Aggregate counts from all threads
    for (int i = 0; i < 10; i++)
    {
        for (int t = 0; t < numThreads; t++)
        {
            count[i] += localCount[t][i];
        }
    }

    // Change count[i] so that count[i] now contains the actual
    // position of this digit in output[]
    for (int i = 1; i < 10; i++)
    {
        count[i] += count[i - 1];
    }

    for (uint64_t i = 0; i < n; i++)
    {
        int digit = (dataArray[n - i - 1] / exp) % 10;
        int idx = count[digit]--;
        output[idx - 1] = dataArray[n - i - 1];
    }

    // Copy the output array to dataArray[], so that dataArray[] now
    // contains sorted numbers according to the current digit
    std::copy(output.begin(), output.end(), dataArray.begin());
}

void radixSort(const vector<int32_t> &dataArray, vector<int32_t> &sortedArray)
{

    // Find the maximum number to know the number of digits
    int32_t m = *std::max_element(dataArray.begin(), dataArray.end());

    // Output array to store sorted numbers temporarily
    std::vector<int32_t> tempArray(dataArray.size());
    std::copy(dataArray.begin(), dataArray.end(), sortedArray.begin());
    // Count array to store the count of occurrences of digits
    std::vector<int> count(10, 0);

    // Do counting sort for every digit. Note that instead
    // of passing the digit number, exp is passed. exp is 10^i
    // where i is the current digit number
    for (uint64_t exp = 1; m / exp > 0; exp *= 10)
    {
        countingSort(sortedArray, exp, tempArray, count);
        std::fill(count.begin(), count.end(), 0);
    }
}

void printArray(const std::vector<int32_t> &dataArray)
{
    for (auto num : dataArray)
    {
        std::cout << num << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        cout << argc << "\n";
        printf("Array size required.\n");
        printf("Syntax: %s <array_size>.\n", argv[0]);
        exit(1);
    }

    uint64_t n = atoll(argv[1]);
    vector<int32_t> dataArray;
    initVector(n, dataArray);
    vector<int32_t> sortedArray(n);
    cout << "Done initializing data\n";

    auto start = std::chrono::high_resolution_clock::now();
    for (int32_t w = 0; w < WARMUP; w++)
    {
        radixSort(dataArray, sortedArray);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedTime = (end - start)/WARMUP;
    cout << "Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;
    // cout << "Original Array:\n";
    // printArray(dataArray);
    // cout << "\n\nSorted array:\n";
    // printArray(sortedArray);
    return 0;
} /* main */
