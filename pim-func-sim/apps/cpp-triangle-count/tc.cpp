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
#include <omp.h>

#include "../util.h"
#include "libpimsim.h"

#define BITS_PER_INT 32

#define NUM_SUBARRAY 4096
#define ROW_SIZE 8192
#define COL_SIZE 8192

uint64_t WORDS_PER_RANK = (uint64_t) 4096 * (uint64_t) 8192 * (uint64_t) 8192 / (uint64_t) BITS_PER_INT;

typedef uint32_t UINT32;


using namespace std;


// Params ---------------------------------------------------------------------
typedef struct Params
{
  uint64_t vectorLength;
  char *configFile;
  char *inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./add [options]"
          "\n"
          "\n    -l    input size (default=8M elements)"
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
  p.vectorLength = 65536;
  p.configFile = nullptr;
  p.inputFile = nullptr;
  p.shouldVerify = false;

  int opt;
  while ((opt = getopt(argc, argv, "h:l:c:i:v:")) >= 0)
  {
    switch (opt)
    {
    case 'h':
      usage();
      exit(0);
      break;
    case 'l':
      p.vectorLength = strtoull(optarg, NULL, 0);
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

int vectorAndPopCntRedSum(uint64_t numElements, std::vector<unsigned int> &src1, std::vector<unsigned int> &src2, std::vector<unsigned int> &dst){
    unsigned bitsPerElement = sizeof(int) * 8;

    PimObjId srcObj1 = pimAlloc(PIM_ALLOC_V1, numElements, bitsPerElement, PIM_INT32);
    if (srcObj1 == -1)
    {
        std::cout << "Abort" << std::endl;
        return -1;
    }

    PimObjId srcObj2 = pimAllocAssociated(bitsPerElement, srcObj1, PIM_INT32);
    if (srcObj2 == -1)
    {
        std::cout << "Abort" << std::endl;
        return -1;
    }

    PimStatus status = pimCopyHostToDevice((void *)src1.data(), srcObj1);
    if (status != PIM_OK)
    {
        std::cout << "Abort" << std::endl;
        return -1;
    }

    status = pimCopyHostToDevice((void *)src2.data(), srcObj2);
    if (status != PIM_OK)
    {
        std::cout << "Abort" << std::endl;
        return -1;
    }
    
    status = pimAnd(srcObj1, srcObj2, srcObj1);
    if (status != PIM_OK)
    {
        std::cout << "Abort" << std::endl;
        return -1;
    }

    status = pimPopCount(srcObj1, srcObj2);
    if (status != PIM_OK)
    {
        std::cout << "Abort" << std::endl;
        return -1;
    }

    int sum = 0;
    status = pimRedSum(srcObj2, &sum);
    if (status != PIM_OK)
    {
        std::cout << "Abort" << std::endl;
        return -1;
    }

    pimFree(srcObj1);
    pimFree(srcObj2);

    return sum;
}

int run_naive(const vector<vector<bool>>& adjMatrix, const vector<vector<UINT32>>& bitAdjMatrix) {
    int count = 0;
    int V = bitAdjMatrix.size();
    // unsigned numElements = V;
    int numElements = (V + BITS_PER_INT - 1) / BITS_PER_INT; // Number of 32-bit integers needed per row
    cout << "number of ndoes: " << V << endl;
    cout << "numElem: " << numElements << endl;
    cout << "WORDS_PER_RANK: " << WORDS_PER_RANK << endl;
    assert(numElements <= (WORDS_PER_RANK / 2) && "Number of vertices cannot exceed (WORDS_PER_RANK / 2)");
    int oneCount = 0;

    #pragma omp parallel for reduction(+:count)
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (adjMatrix[i][j]) { // If there's an edge between i and j
                oneCount++;
                std::vector<unsigned int> src1(numElements);
                std::vector<unsigned int> src2(numElements);
                std::vector<unsigned int> dest(numElements);
                // cout << "i: " << i << ", j: " << j << endl;
                for (int k = 0; k < numElements; ++k) {
                    src1[k] = bitAdjMatrix[i][k];
                    src2[k] = bitAdjMatrix[j][k];
                } 
                int sum = vectorAndPopCntRedSum((uint64_t) numElements, src1, src2, dest);
                if(sum < 0)
                    return -1;
                //redsum
                count += sum;
            }
        }
    }
    cout << "oneCount: " << oneCount << endl;
    cout << "bit32TriangleCount: " << count / 6 << endl;
    // Each triangle is counted three times (once at each vertex), so divide the count by 3
    return count / 6;
}

int run_rowmaxusage(const vector<vector<bool>>& adjMatrix, const vector<vector<UINT32>>& bitAdjMatrix) {
    int count = 0;
    int V = bitAdjMatrix.size();
    uint64_t wordsPerMatrixRow = (V + BITS_PER_INT - 1) / BITS_PER_INT; // Number of 32-bit integers needed per row
    cout << "number of ndoes: " << V << endl;
    cout << "wordsPerMatrixRow: " << wordsPerMatrixRow << endl;
    cout << "WORDS_PER_RANK: " << WORDS_PER_RANK << endl;
    assert(wordsPerMatrixRow <=  (WORDS_PER_RANK / 2) && "Number of vertices cannot exceed (WORDS_PER_RANK / 2)");
    int oneCount = 0;
    uint64_t words = 0;
    std::vector<unsigned int> src1;
    std::vector<unsigned int> src2;
    std::vector<unsigned int> dest;
    int step = V / 10; // Each 10 percent of the total iterations
    uint16_t iterations = 0;
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (adjMatrix[i][j]) { // If there's an edge between i and j
                ++oneCount;
                //TODO: check if we need to add timing for this
                for (int k = 0; k < wordsPerMatrixRow; ++k) {
                    ++words;
                    src1.push_back(bitAdjMatrix[i][k]);
                    src2.push_back(bitAdjMatrix[j][k]);
                }
                if((words < WORDS_PER_RANK / 2) && (words + wordsPerMatrixRow < (WORDS_PER_RANK / 2))){
                    continue;
                }
                cout << "words: " << words << endl;
                iterations++;
                cout << "-------------itr[" << iterations << "]-------------" << endl;
                auto start = std::chrono::high_resolution_clock::now();
                int sum = vectorAndPopCntRedSum((uint64_t) words, src1, src2, dest);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> elapsedTime = (end - start);
                cout << "vectorAndPopCntRedSum time: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;

                if(sum < 0)
                    return -1;
                words = 0;
                src1.clear();
                src2.clear();
                dest.clear();
                count += sum;
            } 
            //last raound
            if((i == V - 1) && (j == V - 1) && words > 0){
                cout << "lr words: " << words << endl;
                cout << "oneCount: " << oneCount << endl;

                auto start = std::chrono::high_resolution_clock::now();
                int sum = vectorAndPopCntRedSum((uint64_t) words, src1, src2, dest);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> elapsedTime = (end - start);
                cout << "vectorAndPopCntRedSum time: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;

                count += sum;
                if(sum < 0)
                    return -1;
                src1.clear();
                src2.clear();
                dest.clear();
            }
        }
        if (i % step == 0) {
            std::cout << "run_rowmaxusage: Progress: " << (i * 100 / V) << "\% completed." << std::endl;
        }
    }
    cout << "oneCount: " << oneCount << endl;
    cout << "bit32TriangleCount: " << count / 6 << endl;
    // Each triangle is counted three times (once at each vertex), so divide the count by 3
    return count / 6;
}

int main(int argc, char** argv) {
    try {
        struct Params params = getInputParams(argc, argv);
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
        cout << "Adjacency Matrix size:" << adjMatrix.size() << endl;

        vector<vector<UINT32>> bitAdjMatrix = convertToBitwiseAdjMatrix(adjMatrix);

        if (!createDevice(params.configFile))
            return 1;
        //run simulation
        // run_naive(adjMatrix, bitAdjMatrix);
        run_rowmaxusage(adjMatrix, bitAdjMatrix);

        //stats
        pimShowStats();

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
    return 0;
}