// Triangle Counting Benchmark
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <bitset>
#include <unordered_set>
#include <sstream>
#include <getopt.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "../../util.h"
#include "libpimeval.h"

#define DEBUG 0

#define BITS_PER_INT 32

typedef uint32_t UINT32;


using namespace std;


// Params ---------------------------------------------------------------------
typedef struct Params
{
  const char *configFile;
  const char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./tc.out [options]"
          "\n"
          "\n    -c    dramsim config file"
          "\n    -i    input file containing two vectors (default=generates vector with random numbers)"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

template <typename T>
void printNestedVector(const std::vector<std::vector<T>>& nestedVec) {
    for (const std::vector<T>& innerVec : nestedVec) {
        for (const T& element : innerVec) {
            std::cout << element << " ";
        }
        std::cout << std::endl; // Print a newline after each inner vector
    }
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.configFile = nullptr;
  p.inputFile = "Dataset/v18772_symetric";
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'c':
      p.configFile = optarg;
      break;
    case 'i':
      p.inputFile = optarg;
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

// Function to convert edge list to adjacency matrix
vector<vector<bool>> edgeListToAdjMatrix(const vector<pair<int, int>>& edgeList, int numNodes) {
    vector<vector<bool>> adjMatrix(numNodes+1, vector<bool>(numNodes+1, 0));

    for (const auto& edge : edgeList) {
        int u = edge.first;
        int v = edge.second;
        adjMatrix[u][v] = adjMatrix[v][u] = 1; // assuming undirected graph
    }

    return adjMatrix;
}

// Function to convert standard adjacency matrix to bitwise adjacency matrix
vector<vector<UINT32>> convertToBitwiseAdjMatrix(const vector<vector<bool>>& adjMatrix) {
    int V = adjMatrix.size();
    int numInts = (V + BITS_PER_INT - 1) / BITS_PER_INT; // Number of 32-bit integers needed per row

    vector<vector<UINT32>> bitAdjMatrix(V, vector<UINT32>(numInts, 0));
    int step = V / 10; // Each 10 percent of the total iterations

    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (adjMatrix[i][j]) {
                bitAdjMatrix[i][j / BITS_PER_INT] |= (1 << (j % BITS_PER_INT));
            }
        }
        if (i % step == 0) {
            std::cout << "convertToBitwiseAdjMatrix: Progress: " << (i * 100 / V) << "\% completed." << std::endl;
        }
    }

    return bitAdjMatrix;
}

// Function to read edge list from a JSON file
vector<pair<int, int>> readEdgeList(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Unable to open file");
    }

    vector<pair<int, int>> edgeList;

    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        int u, v;
        if (!(iss >> u >> v)) {
            throw runtime_error("Invalid file format");
        }
        edgeList.emplace_back(u, v);
    }
    cout << "Edge list size: " << edgeList.size() << endl;
    return edgeList;
}

int vectorAndPopCntRedSum(uint64_t numElements, std::vector<unsigned int> &src1, std::vector<unsigned int> &src2, std::vector<unsigned int> &dst, std::vector<unsigned int> &popCountSrc) {
    unsigned bitsPerElement = sizeof(int) * 8;

    cout << "numElements: " << numElements << endl;

    PimObjId srcObj1 = pimAlloc(PIM_ALLOC_AUTO, numElements, bitsPerElement, PIM_INT32);
    if (srcObj1 == -1)
    {
        std::cout << "src1: pimAlloc" << std::endl;
        return -1;
    }
    if (DEBUG) cout << "src1 allocated successfully!" << endl;

    PimObjId srcObj2 = pimAllocAssociated(bitsPerElement, srcObj1, PIM_INT32);
    if (srcObj2 == -1)
    {
        std::cout << "src2: pimAllocAssociated" << std::endl;
        return -1;
    }
    if (DEBUG) cout << "src2 allocated successfully!" << endl;

    PimObjId dstObj = pimAllocAssociated(bitsPerElement, srcObj1, PIM_INT32);
    if (dstObj == -1)
    {
        std::cout << "dst: pimAllocAssociated" << std::endl;
        return -1;
    }
    if (DEBUG) cout << "dst allocated successfully!" << endl;

    PimObjId popCountSrcObj = pimAllocAssociated(bitsPerElement, srcObj1, PIM_INT32);
    if (popCountSrcObj == -1)
    {
        std::cout << "popCountSrc: pimAllocAssociated" << std::endl;
        return -1;
    }
    if (DEBUG) cout << "popCountSrc allocated successfully!" << endl;

    PimStatus status = pimCopyHostToDevice((void *)src1.data(), srcObj1);
    if (status != PIM_OK)
    {
        std::cout << "src1: pimCopyHostToDevice Abort" << std::endl;
        return -1;
    }
    if (DEBUG) cout << "src1 copied successfully!" << endl;

    status = pimCopyHostToDevice((void *)src2.data(), srcObj2);
    if (status != PIM_OK)
    {
        std::cout << "src2: pimCopyHostToDevice Abort" << std::endl;
        return -1;
    }
    if (DEBUG) cout << "src2 copied successfully!" << endl;

    status = pimAnd(srcObj1, srcObj2, dstObj);
    if (status != PIM_OK)
    {
        std::cout << "pimAnd Abort" << std::endl;
        return -1;
    }
    if (DEBUG) cout << "pimAnd completed successfully!" << endl;

    status = pimPopCount(dstObj, popCountSrcObj);
    if (status != PIM_OK)
    {
        std::cout << "pimPopCount Abort" << std::endl;
        return -1;
    }
    if (DEBUG) cout << "pimPopCount completed successfully!" << endl;

    int64_t sum = 0;
    status = pimRedSumInt(popCountSrcObj, &sum);
    if (status != PIM_OK)
    {
        std::cout << "pimRedSumInt Abort" << std::endl;
        return -1;
    }
    if (DEBUG) cout << "pimRedSum completed successfully!" << endl;

    pimFree(srcObj1);
    pimFree(srcObj2);
    pimFree(dstObj);
    pimFree(popCountSrcObj);

    return sum;
}

int run_rowmaxusage_opt(const vector<vector<bool>>& adjMatrix, const vector<vector<UINT32>>& bitAdjMatrix, uint64_t words_per_device) {
    uint64_t operandsCount = 4; // src1, src2, dst, popCountSrc
    uint64_t operandMaxNumberOfWords = words_per_device / operandsCount;
    int count = 0;
    int V = bitAdjMatrix.size();
    uint64_t wordsPerMatrixRow = (V + BITS_PER_INT - 1) / BITS_PER_INT; // Number of 32-bit integers needed per row
    cout << "wordsPerMatrixRow: " << wordsPerMatrixRow << endl;
    cout << "words_per_device: " << words_per_device << endl;
    assert(wordsPerMatrixRow <=  operandMaxNumberOfWords && "Number of vertices cannot exceed (words_per_device / 2)");
    int oneCount = 0;
    uint64_t words = 0;
    std::vector<unsigned int> src1;
    std::vector<unsigned int> src2;
    int step = V / 10; // Each 10 percent of the total iterations
    uint16_t iterations = 0;
    double host_time_if = 0.0, host_time_forloop = 0.0;
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (adjMatrix[i][j]) 
            { // If there's an edge between i and j
                ++oneCount;
                auto start = std::chrono::high_resolution_clock::now();
#ifdef ENABLE_PARALLEL
                // Parallelizing the loop with OpenMP
                #pragma omp parallel for reduction(+:host_time_if, words) 
#endif
                for (int k = 0; k < wordsPerMatrixRow; ++k) {
                    unsigned int op1 = bitAdjMatrix[i][k];
                    unsigned int op2 = bitAdjMatrix[j][k];
                    auto ifstart = std::chrono::high_resolution_clock::now();
                    if (op1 == 0 || op2 == 0)
                    {
                        auto ifend = std::chrono::high_resolution_clock::now();
                        auto ifelapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(ifend - ifstart);
                        host_time_if += ifelapsedTime.count();
                        continue;
                    }
                    auto ifend = std::chrono::high_resolution_clock::now();
                    auto ifelapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(ifend - ifstart);
                    host_time_if += ifelapsedTime.count();
#ifdef ENABLE_PARALLEL
                    #pragma omp atomic
#endif
                    ++words;
#ifdef ENABLE_PARALLEL
                    #pragma omp critical
#endif
                    {
                        src1.push_back(op1);
                        src2.push_back(op2);
                    }
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                host_time_forloop += elapsedTime.count();
            }
            if((words + wordsPerMatrixRow > operandMaxNumberOfWords) || ((i == V - 1) && (j == V - 1) && words > 0)){
                cout << "words: " << words << endl;
                cout << "-------------itr[" << iterations << "]-------------" << endl;
                std::vector<unsigned int> dst(words);
                std::vector<unsigned int> popCountSrc(words);
                int sum = vectorAndPopCntRedSum((uint64_t) words, src1, src2, dst, popCountSrc);
           
                if(sum < 0)
                    return -1;
                words = 0;
                iterations++;
                src1.clear();
                src2.clear();
                dst.clear();
                count += sum;
            }

        }
        if (i % step == 0) {
            std::cout << "run_rowmaxusage: Progress: " << (i * 100 / V) << "\% rows completed." << std::endl;
        }
    }
    cout << "Host time (for loop): " << std::fixed << std::setprecision(3) << host_time_forloop << " ms." << endl;
    cout << "Host time (if): " << std::fixed << std::setprecision(3) << host_time_if << " ms." << endl;
    cout << "TriangleCount: " << count / 6 << endl;
    // Each triangle is counted 6 times (once at each vertex), so divide the count by 6
    return count / 6;
}

int run_rowmaxusage(const vector<vector<bool>>& adjMatrix, const vector<vector<UINT32>>& bitAdjMatrix, uint64_t words_per_device) {
    uint64_t operandsCount = 4; // src1, src2, dst, popCountSrc
    uint64_t operandMaxNumberOfWords = words_per_device / operandsCount;
    int count = 0;
    int V = bitAdjMatrix.size();
    uint64_t wordsPerMatrixRow = (V + BITS_PER_INT - 1) / BITS_PER_INT; // Number of 32-bit integers needed per row
    assert(wordsPerMatrixRow <=  operandMaxNumberOfWords && "Number of vertices cannot exceed (words_per_device / 2)");
    int oneCount = 0;
    uint64_t words = 0;
    std::vector<unsigned int> src1;
    std::vector<unsigned int> src2;
    int step = V / 10; // Each 10 percent of the total iterations
    uint16_t iterations = 0;

    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (adjMatrix[i][j]) 
            { // If there's an edge between i and j
                ++oneCount;
                for (int k = 0; k < wordsPerMatrixRow; ++k) {
                    unsigned int op1 = bitAdjMatrix[i][k];
                    unsigned int op2 = bitAdjMatrix[j][k];
                    ++words;
                    src1.push_back(op1);
                    src2.push_back(op2);
                }
            }
            if((words + wordsPerMatrixRow > operandMaxNumberOfWords) || ((i == V - 1) && (j == V - 1) && words > 0)){
                cout << "-------------itr[" << iterations << "]-------------" << endl;
                cout << "Number of words that are processed in this iteration: " << words << endl;
                std::vector<unsigned int> dst(words);
                std::vector<unsigned int> popCountSrc(words);
                int sum = vectorAndPopCntRedSum((uint64_t) words, src1, src2, dst, popCountSrc);

                if(sum < 0)
                    return -1;
                
                words = 0;
                iterations++;
                src1.clear();
                src2.clear();
                dst.clear();
                count += sum;
            }

        }
        if (i % step == 0) {
            std::cout << "run_rowmaxusage: Progress: " << (i * 100 / V) << "\% rows completed." << std::endl;
        }
    }
    // Each triangle is counted 6 times (once at each vertex), so divide the count by 6
    return count / 6;
}

int cpuTriangleCount(const vector<vector<bool>>& adjMatrix) {
    int count = 0;
    int V = adjMatrix.size();
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            for (int k = 0; k < V; ++k) {
                if (adjMatrix[i][j] && adjMatrix[j][k] && adjMatrix[k][i]) {
                    ++count;
                }
            }
        }
    }
    return count / 6; // Each triangle is counted 6 times (once at each vertex), so divide the count by 6
}

int main(int argc, char** argv) {
    try {
        struct Params params = getInputParams(argc, argv);
        cout << "Running triangle count on input graph file: " << params.inputFile << endl;
        // Read edge list from JSON file
        vector<pair<int, int>> edgeList = readEdgeList(params.inputFile);
        
        // Determine the number of nodes
        unordered_set<int> nodes;
        for (const auto& edge : edgeList) {
            nodes.insert(edge.first);
            nodes.insert(edge.second);
        }
        int numNodes = nodes.size();
        cout << "Number of nodes: " << numNodes << endl;

        // Convert edge list to adjacency matrix
        vector<vector<bool>> adjMatrix = edgeListToAdjMatrix(edgeList, numNodes);
        cout << "-----------convertToBitwiseAdjMatrix-----------" << endl;
    
        vector<vector<UINT32>> bitAdjMatrix = convertToBitwiseAdjMatrix(adjMatrix);

        cout << "-----------createDevice-----------" << endl;
        // create device
        if (!createDevice(params.configFile))
            return 1;

        // get device parameters 
        PimDeviceProperties deviceProp;
        PimStatus status = pimGetDeviceProperties(&deviceProp);
        assert(status == PIM_OK);

        uint64_t words_per_device = (uint64_t) deviceProp.numRanks * deviceProp.numBankPerRank * deviceProp.numSubarrayPerBank * deviceProp.numRowPerSubarray * deviceProp.numColPerSubarray / BITS_PER_INT;
        //run simulation
        cout << "-----------Start running on PIM-----------" << endl;
        int pimTriCount = run_rowmaxusage(adjMatrix, bitAdjMatrix, words_per_device);

        if (params.shouldVerify){
            //run on cpu
            cout << "-----------Triangle Count Verification-----------" << endl;
            int cpuTriCount = cpuTriangleCount(adjMatrix);
            if (cpuTriCount != pimTriCount)
                printf("PIM count (%d) does not match CPU count(%d)\n", pimTriCount, cpuTriCount);
            else
                printf("PIM count (%d) matches CPU count(%d)\n", pimTriCount, cpuTriCount);
        }

        //stats
        pimShowStats();

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
    return 0;
}
