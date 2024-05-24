#include <iostream>
#include <vector>
#include <cassert>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <unordered_set>
#include <sstream>
#include <bitset>


typedef uint32_t UINT32;
const int BITS_PER_INT = 32;

using namespace std;

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

// Function to convert edge list to adjacency matrix
vector<vector<int>> edgeListToAdjMatrix(const vector<pair<int, int>>& edgeList, int numNodes) {
    vector<vector<int>> adjMatrix(numNodes+1, vector<int>(numNodes+1, 0));

    for (const auto& edge : edgeList) {
        int u = edge.first;
        int v = edge.second;
        adjMatrix[u][v] = adjMatrix[v][u] = 1; // assuming undirected graph
    }

    return adjMatrix;
}

int countTrianglesMultiThread(const vector<vector<int>>& adjMatrix) {
    int count = 0;
    int V = adjMatrix.size(); // Number of vertices
    #pragma omp parallel for reduction(+:count)
    for (int i = 0; i < V; ++i) {
        for (int j = i + 1; j < V; ++j) {
            if (adjMatrix[i][j]) { // Check if there is an edge between i and j
                for (int k = j + 1; k < V; ++k) {
                    if (adjMatrix[j][k] && adjMatrix[i][k]) { // Check for the other two edges
                        count++;
                    }
                }
            }
        }
    }
    cout << "OMP Number of triangles: " << count << endl;
    return count;
}

// Function to count triangles in an undirected graph using the traditional method
int countTriangles(const vector<vector<int>>& adjMatrix) {
    int V = adjMatrix.size(); // Number of vertices
    int count = 0;

    // Consider every possible triplet of vertices (i, j, k)
    for (int i = 0; i < V; ++i) {
        for (int j = i + 1; j < V; ++j) {
            if (adjMatrix[i][j]) { // Check if there is an edge between i and j
                for (int k = j + 1; k < V; ++k) {
                    if (adjMatrix[j][k] && adjMatrix[i][k]) { // Check for the other two edges
                        count++;
                    }
                }
            }
        }
    }
    cout << "Single Number of Triangles: " << count << endl;
    return count;
}

// Function to count triangles using the row and column dot product method
int bitTriangleCount(const vector<vector<int>>& adjMatrix) {
    int V = adjMatrix.size(); // Number of vertices
    int count = 0;

    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (adjMatrix[i][j]) { // If there's an edge between i and j
                // Calculate the dot product of row i and row j
                int dotProduct = 0;
                for (int k = 0; k < V; ++k) {
                    dotProduct += adjMatrix[i][k] * adjMatrix[j][k];
                }
                count += dotProduct;
            }
        }
    }
    cout << "bitTriangleCount: " << count/6 << endl;
    // Each triangle is counted three times (once at each vertex), so divide the count by 3
    return count / 6;
}

// Function to convert standard adjacency matrix to bitwise adjacency matrix
vector<vector<UINT32>> convertToBitwiseAdjMatrix(const vector<vector<int>>& adjMatrix) {
    int V = adjMatrix.size();
    int numInts = (V + BITS_PER_INT - 1) / BITS_PER_INT; // Number of 32-bit integers needed per row

    vector<vector<UINT32>> bitAdjMatrix(V, vector<UINT32>(numInts, 0));

    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (adjMatrix[i][j]) {
                bitAdjMatrix[i][j / BITS_PER_INT] |= (1 << (j % BITS_PER_INT));
            }
        }
    }

    return bitAdjMatrix;
}

// Function to count triangles using bitwise AND operation
int bit32TriangleCount(const vector<vector<int>>& adjMatrix, const vector<vector<UINT32>>& bitAdjMatrix) {
    int count = 0;
    int V = bitAdjMatrix.size();
    int numInts = (V + BITS_PER_INT - 1) / BITS_PER_INT; // Number of 32-bit integers needed per row

    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (adjMatrix[i][j]) { // If there's an edge between i and j
                // int l = j / BITS_PER_INT;
                int dotProduct = 0;
                for (int k = 0; k < numInts; ++k) {
                    dotProduct += __builtin_popcount(bitAdjMatrix[i][k] & bitAdjMatrix[j][k]);
                }
                count += dotProduct;
            }
        }
    }

    cout << "bit32TriangleCount: " << count / 6 << endl;
    // Each triangle is counted three times (once at each vertex), so divide the count by 3
    return count / 6;
}


// Function to run unit tests
void runTests() {
    // Test 1: Simple triangle
    // vector<vector<int>> adjMatrix1 = {
    //     {0, 1, 1, 0},
    //     {1, 0, 1, 0},
    //     {1, 1, 0, 0},
    //     {0, 0, 0, 0}
    // };
    // assert(countTriangles(adjMatrix1) == 1);
    // assert(bitTriangleCount(adjMatrix1) == 1);

    // // Test 2: No triangles
    // vector<vector<int>> adjMatrix2 = {
    //     {0, 1, 0, 0},
    //     {1, 0, 1, 0},
    //     {0, 1, 0, 1},
    //     {0, 0, 1, 0}
    // };
    // assert(countTriangles(adjMatrix2) == 0);
    // assert(bitTriangleCount(adjMatrix2) == 0);

    // // Test 3: Multiple triangles
    // vector<vector<int>> adjMatrix3 = {
    //     {0, 1, 1, 1},
    //     {1, 0, 1, 1},
    //     {1, 1, 0, 1},
    //     {1, 1, 1, 0}
    // };
    // assert(countTriangles(adjMatrix3) == 4);
    // assert(bitTriangleCount(adjMatrix3) == 4);

    // Test 4: Larger graph with 2 triangles
    cout << "Test 4: Large sparse graph\n";
    vector<vector<int>> adjMatrix4 = {
        {0, 1, 1, 1, 0},
        {1, 0, 1, 0, 0},
        {1, 1, 0, 0, 0},
        {1, 0, 0, 0, 1},
        {0, 0, 0, 1, 0}
    };
    assert(countTriangles(adjMatrix4) == 1);
    assert(bitTriangleCount(adjMatrix4) == 1);

    // Test 5: Large sparse graph (10 vertices, 1 triangle)
    cout << "Test 5: Large sparse graph\n";
    vector<vector<int>> adjMatrix5 = {
        {0, 1, 1, 0, 0, 0, 0, 0, 0, 0},
        {1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
        {1, 1, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 1, 1, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 1, 0, 0, 0, 0},
        {0, 0, 0, 1, 1, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 1, 1, 1},
        {0, 0, 0, 0, 0, 0, 1, 0, 1, 0},
        {0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
        {0, 0, 0, 0, 0, 0, 1, 0, 0, 0}
    };
    
    assert(countTriangles(adjMatrix5) == 3);
    assert(bitTriangleCount(adjMatrix5) == 3);
    
    cout << "All tests passed!" << endl;
}

// Function to print adjacency matrix
void printAdjMatrix(const vector<vector<int>>& adjMatrix) {
    for (const auto& row : adjMatrix) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
}

void run(const vector<vector<int>>& adjMatrix){
    cout << "------------optimized baseline------------" << endl;
    auto start = std::chrono::high_resolution_clock::now();
    countTriangles(adjMatrix);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsedTime = (end - start);
    cout << "Baseline Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;
    
    cout << "------------bit32 (1b per word)------------" << endl;
    cout << "Running optimized..." << endl;
    start = std::chrono::high_resolution_clock::now();
    bitTriangleCount(adjMatrix);
    end = std::chrono::high_resolution_clock::now();
    elapsedTime = (end - start);
    cout << "bit32 Basline Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;

    cout << "------------bit32 (32b per word)------------" << endl;
    vector<vector<UINT32>> bitAdjMatrix = convertToBitwiseAdjMatrix(adjMatrix);
    start = std::chrono::high_resolution_clock::now();
    bit32TriangleCount(adjMatrix, bitAdjMatrix);
    end = std::chrono::high_resolution_clock::now();
    elapsedTime = (end - start);
    cout << "bit32 Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;

    cout << "------------MultiThreaded------------" << endl;
    start = std::chrono::high_resolution_clock::now();
    countTrianglesMultiThread(adjMatrix);
    end = std::chrono::high_resolution_clock::now();
    elapsedTime = (end - start);
    cout << "OMP Duration: " << std::fixed << std::setprecision(3) << elapsedTime.count() << " ms." << endl;

}

int main(int argc, char** argv) {
     try {
        // Read edge list from JSON file
        string filename = argv[1];
        vector<pair<int, int>> edgeList = readEdgeList(filename);

        // Determine the number of nodes
        unordered_set<int> nodes;
        for (const auto& edge : edgeList) {
            nodes.insert(edge.first);
            nodes.insert(edge.second);
        }
        int numNodes = nodes.size();
        cout << "Number of nodes: " << numNodes << endl;

        // Convert edge list to adjacency matrix
        vector<vector<int>> adjMatrix = edgeListToAdjMatrix(edgeList, numNodes);
        cout << "Adjacency Matrix size:" << adjMatrix.size() << endl;

        //run
        run(adjMatrix);

        // Print the adjacency matrix
        // cout << "Adjacency Matrix:" << endl;
        // printAdjMatrix(adjMatrix);
        // runTests();
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
    return 0;
}