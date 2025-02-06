#ifndef PIM_FUNC_SIM_BASELINE_UTIL_H
#define PIM_FUNC_SIM_BASELINE_UTIL_H

#include <iostream>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <omp.h>
#include <cstdint>

using namespace std;

#define WARMUP 2
#define MAX_NUMBER 1024

/**
 * @brief Initializes a vector with random values.
 *
 * This function creates a vector of the specified size and fills it with random
 * values ranging from 0 to MAX_NUMBER. It uses parallelism to speed up the initialization.
 *
 * @param vectorSize The size of the vector to be created.
 * @param dataVector A reference to the vector that will be initialized.
 */
template <typename T>
void getVector(uint64_t vectorSize, std::vector<T>& dataVector) {
    // Resize the vector to the specified size
    dataVector.resize(vectorSize);

    // Seed the random number generator with a fixed seed for reproducibility
    srand(8746219);

    #pragma omp parallel for
    for (uint64_t i = 0; i < vectorSize; ++i) {
        dataVector[i] = static_cast<T>(rand() % MAX_NUMBER);
    }
}

/**
 * @brief Initializes a matrix with random values.
 *
 * This function creates a matrix with the specified number of rows and columns
 * and fills it with random values ranging from 0 to MAX_NUMBER. It uses parallelism
 * to speed up the initialization.
 *
 * @param numRows The number of rows in the matrix.
 * @param numCols The number of columns in the matrix.
 * @param matrix A reference to the matrix that will be initialized.
 */
template <typename T>
void getMatrix(uint64_t numRows, uint64_t numCols, vector<vector<T>> &matrix) {

    // Seed the random number generator with a fixed seed for reproducibility
    srand(8746219);

    // Resize the matrix to the specified number of rows and columns
    matrix.resize(numRows, vector<T>(numCols));

    #pragma omp parallel for
    for (uint64_t row = 0; row < numRows; ++row) {
        for (uint64_t col = 0; col < numCols; ++col) {
            matrix[row][col] = rand() % MAX_NUMBER;
        }
    }
}

#endif // _COMMON_H_
