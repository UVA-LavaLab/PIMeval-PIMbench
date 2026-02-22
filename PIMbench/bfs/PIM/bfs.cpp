// Test: C++ version of breadth-first search
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include <iostream>
#include <vector>
#include <getopt.h>
#include <stdint.h>
#include <iomanip>
#include <fstream>
#include <string>
#include <unordered_map>
#if defined(_OPENMP)
#include <omp.h>
#endif

#include "util.h"
#include "libpimeval.h"
#include <queue>

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
  char *configFile;
  std::string inputFile;
  bool shouldVerify;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./bfs.out [options]"
          "\n"
          "\n    -i    text file (.txt) containing the edge list (default=../dataset/email-Eu-core.txt)"
          "\n    -c    dramsim config file"
          "\n    -v    t = verifies PIM output with host output. (default=false)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.configFile = nullptr;
  p.inputFile = "../dataset/email-Eu-core.txt";
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

void createCSR(ifstream &fin, vector<uint> &rowIDList, vector<uint> &colIDList, uint64_t &numVertices, uint64_t &numEdges) {
  uint64_t u = 0, v = 0;
  uint64_t maxId = 0;
  std::vector<uint64_t> deg;
  numEdges = 0;
  while (fin >> u >> v) {
    if (u > maxId) maxId = u;
    if (v > maxId) maxId = v;

    if (maxId >= deg.size()) {
      deg.resize(maxId + 1, 0);
    }
    deg[u] += 1;
    numEdges += 1;
  }
  numVertices = maxId + 1;
  rowIDList.assign(numVertices + 1, 0);
  for (uint64_t i = 0; i < numVertices; ++i) {
    rowIDList[i + 1] = rowIDList[i] + deg[i];
  }
  colIDList.assign(numEdges, 0);
  std::vector<uint> next = rowIDList;
  fin.clear();                 // this is important to clear EOF/fail flags
  fin.seekg(0, std::ios::beg); // rewind to beginning of file
  while (fin >> u >> v) {
    colIDList[next[u]++] = v;
  }
}

void pinVerticesToCores(uint64_t numVertices, PimDeviceProperties &deviceProps, PimObjId &vertexObj, PimObjId &startMaskObj, PimObjId &endMaskObj, std::vector<int> &vertexVector, uint64_t elementsPerRow)
{
    // Let's assume vector Length = #vertices
  uint64_t verticesPerCore = std::ceil(numVertices * 1.0 / deviceProps.numPIMCores);

  // We are pinning vertices to PIM cores such that each core gets contiguous vertices
  // In this way, each core gets equal number of vertices (except maybe the last core which can get fewer vertices if numVertices is not perfectly divisible by numPIMCores)
  // However, we understand that this might not be the most optimal load balancing considering degree distribution of vertices
  // To pin vertices we are creating a vector of size (elementsPerRow * numPIMCores) where each row corresponds to a core and contains the ([startVertex, endVertex)) vertices assigned to that core. 
  // So first two elements in each row indicate the start and end vertex assigned to that core, and the rest of the elements in that row are padded with -1. For example, if we have 4 cores and 10 vertices, and elementsPerRow is 4, then the vertexVector will look like this:
  // Core 0: [0, 3, -1, -1]
  // Core 1: [3, 6, -1, -1]
  // Core 2: [6, 9, -1, -1]
  // Core 3: [9, 10, -1, -1]
  // The rest of the elements in the row are padded with -1. 
  // We also create two mask vectors to indicate the start and end of valid vertices for each core. This is needed because we are padding the vertex vector with -1, and we need a way to differentiate between valid vertices and padded vertices. 
  for (unsigned coreId = 0; coreId < deviceProps.numPIMCores; ++coreId)
  {
    int startVertex = coreId * verticesPerCore;
    int endVertex = std::min(startVertex + verticesPerCore, numVertices);
    vertexVector[coreId * elementsPerRow] = startVertex;
    vertexVector[coreId * elementsPerRow + 1] = endVertex;
  }

  PimStatus status = pimCopyHostToDevice((void *)vertexVector.data(), vertexObj);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  std::vector<uint8_t> startMask(vertexVector.size(), 0);
  std::vector<uint8_t> endMask(vertexVector.size(), 0);
  for (unsigned coreId = 0; coreId < deviceProps.numPIMCores; ++coreId) {
    uint64_t base = (uint64_t)coreId * elementsPerRow;
    startMask[base + 0] = 1;
    endMask[base + 1] = 1;
  }

  status = pimCopyHostToDevice((void *)startMask.data(), startMaskObj);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimCopyHostToDevice((void *)endMask.data(), endMaskObj);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }
}

void allocateRowIndices(PimDeviceProperties &deviceProps, uint64_t elementsPerRow, uint64_t numVertices, uint64_t verticesPerCore, std::vector<uint> &rowIdxVector, std::vector<uint> &pimRowIDVector, PimObjId &rowIdxObj) {
  for (uint64_t coreId = 0; coreId < deviceProps.numPIMCores; ++coreId) {
    uint64_t startV = coreId * verticesPerCore;
    if (startV >= numVertices) break;
    uint64_t endV = std::min(startV + verticesPerCore, numVertices);

    uint64_t localLen = (endV - startV) + 1; // #row_ptr entries for this core

    uint32_t baseOffset = rowIdxVector[startV];

    for (uint64_t i = 0; i < localLen; ++i) {
      uint64_t row  = i / elementsPerRow;
      uint64_t lane = i % elementsPerRow;

      uint64_t idx = ((row * deviceProps.numPIMCores + coreId) * elementsPerRow + lane);

      pimRowIDVector[idx] = (uint32_t)(rowIdxVector[startV + i] - baseOffset);
    }
  }

  PimStatus status = pimCopyHostToDevice((void *)pimRowIDVector.data(), rowIdxObj);
  if (status != PIM_OK)  {
    std::cout << "Abort" << std::endl;
    return;
  }
}

void allocateColumnIndices(PimDeviceProperties &deviceProps, uint64_t elementsPerRow, uint64_t numVertices, uint64_t verticesPerCore, std::vector<uint> &rowIdxVector, std::vector<uint> &colIDList, std::vector<uint> &pimColIDVector, PimObjId &colIdxObj) {
  for (uint64_t coreId = 0; coreId < deviceProps.numPIMCores; ++coreId) {
    uint64_t startV = coreId * verticesPerCore;
    if (startV >= numVertices) break;
    uint64_t endV = std::min(startV + verticesPerCore, numVertices);

    uint64_t startE = (uint64_t)rowIdxVector[startV];
    uint64_t endE   = (uint64_t)rowIdxVector[endV];
    uint64_t nnzCore = endE - startE;

    // copy nnzCore entries into this core's block in striped form
    for (uint64_t i = 0; i < nnzCore; ++i) {
      uint64_t row  = i / elementsPerRow;
      uint64_t lane = i % elementsPerRow;

      // striped index: row0core0,row0core1,...,row1core0,...
      uint64_t idx = ((row * deviceProps.numPIMCores + coreId) * elementsPerRow + lane);

      pimColIDVector[idx] = colIDList[startE + i];
    }

  }

  PimStatus status = pimCopyHostToDevice((void*)pimColIDVector.data(), colIdxObj);
  if (status != PIM_OK) {
    std::cout << "Abort copying colIdx\n";
    return;
  }
}

void runBFS(uint64_t numVertices, std::vector<uint> &rowIDList, std::vector<uint> &colIDList)
{
  PimDeviceProperties deviceProps;
  PimStatus status = pimGetDeviceProperties(&deviceProps);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  uint64_t verticesPerCore = std::ceil(numVertices * 1.0 / deviceProps.numPIMCores);
  uint64_t elementsPerRow = deviceProps.numColPerSubarray / (sizeof(int) * 8);
  uint64_t rowsNeededforRowIndices = std::ceil((verticesPerCore + 1) * 1.0 / elementsPerRow);

  uint64_t maxNnzPerCore = 0;

  for (uint64_t coreId = 0; coreId < deviceProps.numPIMCores; ++coreId) {
    uint64_t startV = coreId * verticesPerCore;
    if (startV >= numVertices) break;
    uint64_t endV = std::min(startV + verticesPerCore, numVertices);

    uint64_t nnzCore = (uint64_t)rowIDList[endV] - (uint64_t)rowIDList[startV];
    if (nnzCore > maxNnzPerCore) maxNnzPerCore = nnzCore;
  }

  uint64_t rowsNeededforColumnIndices =
      (maxNnzPerCore + elementsPerRow - 1) / elementsPerRow;

  uint64_t maxRowsNeeded = std::max(rowsNeededforRowIndices, rowsNeededforColumnIndices);
  maxRowsNeeded = deviceProps.isHLayoutDevice ? maxRowsNeeded : maxRowsNeeded * (sizeof(int) * 8);
  std::cout << "max non zero per core: " << maxNnzPerCore << ", maxRowsNeeded for graph data structure: " << maxRowsNeeded << "\n";

  if (maxRowsNeeded > deviceProps.numRowPerCore) {
    std::cout << "Abort because the graph is too large to fit in the PIM device." << std::endl;
    return;
  }

  std::vector<int> vertexVector(elementsPerRow * maxRowsNeeded * deviceProps.numPIMCores, -1);
  std::vector<uint> pimRowIDVector(elementsPerRow * maxRowsNeeded * deviceProps.numPIMCores, 0);
  std::vector<uint> pimColIDVector(elementsPerRow * maxRowsNeeded * deviceProps.numPIMCores, 0);
  std::vector<int> pimRowIdMaskVector(elementsPerRow * maxRowsNeeded * deviceProps.numPIMCores, 0);
  
  PimObjId vertexObj = pimAlloc(PIM_ALLOC_AUTO, elementsPerRow * maxRowsNeeded * deviceProps.numPIMCores, PIM_INT32);
  if (vertexObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId rowIdxObj = pimAllocAssociated(vertexObj, PIM_UINT32);
  if (rowIdxObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  } 

  PimObjId colIdxObj = pimAllocAssociated(vertexObj, PIM_UINT32);
  if (colIdxObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId startMaskObj = pimAllocAssociated(vertexObj, PIM_BOOL);
  if (startMaskObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId endMaskObj = pimAllocAssociated(vertexObj, PIM_BOOL);
  if (endMaskObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId rowIdMaskObj = pimAllocAssociated(vertexObj, PIM_INT32);
  if (rowIdMaskObj == -1)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  pinVerticesToCores(numVertices, deviceProps, vertexObj, startMaskObj, endMaskObj, vertexVector, elementsPerRow);
  std::cout << "Pinned vertices to cores. Each core gets " << verticesPerCore << " vertices (padded with -1 if needed)." << "\n";
  allocateRowIndices(deviceProps, elementsPerRow, numVertices, verticesPerCore, rowIDList, pimRowIDVector, rowIdxObj);
  allocateColumnIndices(deviceProps, elementsPerRow, numVertices, verticesPerCore, rowIDList, colIDList, pimColIDVector, colIdxObj);
  std::cout << "Moved graph data structure to PIM device.\n Starting BFS traversal from vertex 0...\n\n";

  status = pimCopyHostToDevice((void *)pimRowIdMaskVector.data(), rowIdMaskObj);
  if (status != PIM_OK)
  {
    std::cout << "Abort copying rowIdMaskVector to device" << std::endl;
    return;
  }

  //host maintains the visited information; everytime each PIM core sends the neighbor list, host checks if visisted and updates frontier as well as visited vector
  std::vector<uint8_t> visitedVector(vertexVector.size(), 0);

  int sourceVertex = 0;
  std::queue<int> bfsQueue;
  bfsQueue.push(sourceVertex);
  while(!bfsQueue.empty()) {
    int currVertex = bfsQueue.front();
    bfsQueue.pop();
    PimObjId matchStart = pimAllocAssociated(vertexObj, PIM_BOOL);
    if (matchStart == -1)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    PimObjId matchEnd = pimAllocAssociated(vertexObj, PIM_BOOL);
    if (matchEnd == -1)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimLTScalar(vertexObj, matchStart, currVertex+1, 0, 1, PIM_LOCAL);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimGTScalar(vertexObj, matchEnd, currVertex, 1, 2, PIM_LOCAL);
    if (status != PIM_OK)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimAnd(matchStart, startMaskObj, matchStart);
    status = pimAnd(matchEnd, endMaskObj, matchEnd);
    status = pimShiftElementsLeft(matchEnd);
    status = pimAnd(matchStart, matchEnd, matchStart);

    PimObjId rowIDxOffsetObject = pimAllocAssociated(vertexObj, PIM_INT32);
    if (rowIDxOffsetObject == -1)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimBroadcastInt(rowIDxOffsetObject, currVertex);
    if (status != PIM_OK)
    {     
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimSub(rowIDxOffsetObject, vertexObj, rowIDxOffsetObject, 0, 1, PIM_LOCAL);
    if (status != PIM_OK)
    {     
      std::cout << "Abort" << std::endl;
      return;
    }

    std::vector<int> offsetVector(vertexVector.size(), 0);
    status = pimCopyDeviceToHost(rowIDxOffsetObject, (void *)offsetVector.data());
    if (status != PIM_OK)
    {
      std::cout << "Abort copying rowIDxOffsetObject to host" << std::endl;
      return;
    }

    std::vector<uint8_t> resultVec(vertexVector.size());

    status = pimCopyDeviceToHost(matchStart, (void *)resultVec.data());
    if (status != PIM_OK)
    {
      std::cout << "Abort copying matchStart to host" << std::endl;
      return;
    }

    uint64_t offsetAddress = 0;
    unsigned currCore = 0;

    for (unsigned coreId = 0; coreId < deviceProps.numPIMCores; ++coreId) {
      uint64_t base = (uint64_t)coreId * elementsPerRow;
      if (resultVec[base]) {
        std::cout << "Vertex " << currVertex << " is assigned to core " << coreId << "\n";
        offsetAddress = base;
        currCore = coreId;
        break;
      }
    }

    std::cout << "Core ID holds Vertices in the range: [" << vertexVector[offsetAddress] << ", " << vertexVector[offsetAddress + 1] << ")\n";

    uint64_t off = offsetVector[offsetAddress];
    uint64_t row  = off / elementsPerRow;
    uint64_t lane = off % elementsPerRow;
    uint64_t idx0 = ((row * deviceProps.numPIMCores + currCore) * elementsPerRow + lane);
    // std::cout << "Offset value for current vertex: " << off << ", which corresponds to row " << row << " and lane " << lane << " in the vertex vector.\n";
    // std::cout << "Row index for current vertex's neighbors starts at: " << idx0 << "\n";
    // std::cout << "Actual Row index for current vertex's neighbors starts at: " << pimRowIDVector[idx0] << "\n";
    uint64_t off1  = off + 1;
    uint64_t row1  = off1 / elementsPerRow;
    uint64_t lane1 = off1 % elementsPerRow;
    uint64_t idx1  = ((row1 * deviceProps.numPIMCores + currCore) * elementsPerRow + lane1);
    // std::cout << "Offset value for next vertex: " << off1 << ", which corresponds to row " << row1 << " and lane " << lane1 << " in the vertex vector.\n";
    // std::cout << "Row index for next vertex's neighbors starts at: " << idx1 << "\n";
    // std::cout << "Actual Row index for next vertex's neighbors starts at: " << pimRowIDVector[idx1] << "\n";

    resultVec.assign(vertexVector.size(), 0);
    resultVec[idx0] = 1;
    status = pimCopyHostToDevice((void *)resultVec.data(), matchEnd);
    if (status != PIM_OK)    {
      std::cout << "Abort copying resultVec to device" << std::endl;
      return;
    }

    pimFree(rowIDxOffsetObject);

    rowIDxOffsetObject = pimAllocAssociated(vertexObj, PIM_UINT32);
    if (rowIDxOffsetObject == -1)
    {
      std::cout << "Abort" << std::endl;
      return;
    }

    std::vector<uint32_t> rowI(vertexVector.size(), 0);

    status = pimCopyDeviceToHost(rowIdxObj, (void *)rowI.data());
    if (status != PIM_OK)    {
      std::cout << "Abort copying rowIdxObj to host" << std::endl;
      return;
    }
    // std::cout << "RowIdx for current vertex in PIM: " << rowI[idx0] << ", in host: " << pimRowIDVector[idx0] << ", original CSR: " << rowIDList[currVertex] << "\n";
    // std::cout << "RowIdx for next vertex in PIM: " << rowI[idx1] << ", in host: " << pimRowIDVector[idx1] << ", original CSR: " << rowIDList[currVertex + 1] << "\n";

    status = pimBroadcastUInt(rowIDxOffsetObject, 0);
    if (status != PIM_OK)
    {     
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimCondCopy(matchEnd, rowIdxObj, rowIDxOffsetObject);
    if (status != PIM_OK)    {
      std::cout << "Abort copying rowIdxObj to rowIDxOffsetObject with condition" << std::endl;
      return;
    }

    std::vector<uint32_t> neighborOffsetVector(vertexVector.size(), 0);
    status = pimCopyDeviceToHost(rowIDxOffsetObject, (void *)neighborOffsetVector.data());
    if (status != PIM_OK)    {
      std::cout << "Abort copying rowIDxOffsetObject to host" << std::endl;
      return;
    }

    int beg = neighborOffsetVector[idx0];

    resultVec.assign(vertexVector.size(), 0);
    resultVec[idx1] = 1;
    
    status = pimCopyHostToDevice((void *)resultVec.data(), matchEnd);
    if (status != PIM_OK)    {
      std::cout << "Abort copying resultVec to device" << std::endl;
      return;
    }

    status = pimBroadcastUInt(rowIDxOffsetObject, 0);
    if (status != PIM_OK)
    {     
      std::cout << "Abort" << std::endl;
      return;
    }

    status = pimCondCopy(matchEnd, rowIdxObj, rowIDxOffsetObject);
    if (status != PIM_OK)    {
      std::cout << "Abort copying rowIdxObj to rowIDxOffsetObject with condition" << std::endl;
      return;
    }

    status = pimCopyDeviceToHost(rowIDxOffsetObject, (void *)neighborOffsetVector.data());
    if (status != PIM_OK)    {
      std::cout << "Abort copying rowIDxOffsetObject to host" << std::endl;
      return;
    }

    int end = neighborOffsetVector[idx1];
    
    std::cout << "Current vertex: " << currVertex << ", offset address: " << offsetAddress << ", offset value: " << offsetVector[offsetAddress] << "\n";
    // std::cout << "Row indices for current vertex's neighbors are in the range: [" << beg << ", " << end << ")\n";
    // std::cout << "Actual Row indices for current vertex's neighbors: " << pimRowIDVector[idx0] << " to " << pimRowIDVector[idx1] << "\n";

    std::vector<uint8_t> nbrMaskVec(pimColIDVector.size(), 0);

    for (uint32_t j = beg; j < end; ++j) {
      uint64_t r = j / elementsPerRow;
      uint64_t l = j % elementsPerRow;
      uint64_t idx = ((r * deviceProps.numPIMCores + currCore) * elementsPerRow + l);
      nbrMaskVec[idx] = 1;
    }

    PimObjId nbrMask = pimAllocAssociated(vertexObj, PIM_BOOL);
    pimCopyHostToDevice(nbrMaskVec.data(), nbrMask);

    PimObjId nbrOut = pimAllocAssociated(vertexObj, PIM_UINT32);
    pimBroadcastUInt(nbrOut, 0);
    pimCondCopy(nbrMask, colIdxObj, nbrOut);

    std::vector<uint32_t> neighborIDVector(vertexVector.size(), 0);

    pimCopyDeviceToHost(nbrOut, (void *)neighborIDVector.data());
    // std::cout << "Neighbor IDs for current vertex: ";
    for (uint32_t j = beg; j < end; ++j) {
      uint64_t r = j / elementsPerRow;
      uint64_t l = j % elementsPerRow;
      uint64_t idx = ((r * deviceProps.numPIMCores + currCore) * elementsPerRow + l);
      if (visitedVector[neighborIDVector[idx]]) continue; // if neighbor has been visited, skip
      else {
        visitedVector[neighborIDVector[idx]] = 1; // mark neighbor as visited
        bfsQueue.push(neighborIDVector[idx]); // add neighbor to BFS queue
      }
      //std::cout << neighborIDVector[idx] << " ";
      // Here we can add the logic to check if the neighbor has been visited before (using a visited mask), and if not, add it to the BFS queue and mark it as visited.
    }
    // std::cout << "\n";

    // std::cout << "Actual Neighbor IDs for current vertex: ";
    // for (uint32_t j = rowIDList[currVertex]; j < rowIDList[currVertex + 1]; ++j) {
    //   std::cout << colIDList[j] << " ";
    // }
    // std::cout << "\n";

    pimFree(rowIDxOffsetObject);
    pimFree(matchStart);
    pimFree(matchEnd);
    pimFree(nbrMask);
    pimFree(nbrOut);
  }

  pimFree(vertexObj);
  pimFree(rowIdxObj);
  pimFree(colIdxObj);
  pimFree(startMaskObj);
  pimFree(endMaskObj);
  pimFree(rowIdMaskObj);
}

int main(int argc, char* argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::cout << "Running BFS on PIM. Input is the file:  " << params.inputFile << " containing the edge list." << "\n\n";
  
  // Openning the input file containing the edge list
  std::ifstream fin(params.inputFile);
  if (!fin) {
    std::cerr << "Error opening input file: " << params.inputFile << std::endl;
    return 1;
  }

  uint64_t numVertices = 0;
  uint64_t numEdges = 0;
  std::vector<uint> rowIDList; // CSR row pointer array
  std::vector<uint> colIDList; // CSR column index array
  // Read the edge list from the input file and construct CSR representation of the graph, keep track of the #vertices and #edges in the graph
  createCSR(fin, rowIDList, colIDList, numVertices, numEdges);
  fin.close();
  std::cout << "Created CSR. Graph has " << numVertices << " vertices and " << numEdges << " edges.\n";

  if (!createDevice(params.configFile)) return 1;

  runBFS(numVertices, rowIDList, colIDList);
  if (params.shouldVerify) {
    // verify result
  }

  pimShowStats();

  return 0;
}
