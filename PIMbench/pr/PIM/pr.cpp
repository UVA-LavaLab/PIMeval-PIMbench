// Test: C++ version of page rank
// Copyright (c) 2026 University of Virginia
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

std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();

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
          "\nUsage:  ./pr.out [options]"
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
  p.inputFile = "../dataset/email-Eu-core.el";
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

void createCSC(ifstream &fin, vector<uint64_t> &outDeg, vector<uint> &rowIDList, vector<uint> &colIDList, uint64_t &numVertices, uint64_t &numEdges) {
  uint64_t u = 0, v = 0;
  uint64_t maxId = 0;
  std::vector<uint> inDeg;
  numEdges = 0;
  while (fin >> u >> v) {
    if (u > maxId) maxId = u;
    if (v > maxId) maxId = v;

    if (maxId >= inDeg.size()) {
      inDeg.resize(maxId + 1, 0);
      outDeg.resize(maxId + 1, 0);
    }
    inDeg[v] += 1;
    outDeg[u] += 1;
    numEdges += 1;
  }
  numVertices = maxId + 1;
  colIDList.assign(numVertices + 1, 0);
  for (uint64_t i = 0; i < numVertices; ++i) {
    colIDList[i + 1] = colIDList[i] + inDeg[i];
  }
  std::vector<uint> next = colIDList;
  rowIDList.assign(numEdges, 0);
  fin.clear();                 // this is important to clear EOF/fail flags
  fin.seekg(0, std::ios::beg); // rewind to beginning of file
  while (fin >> u >> v) {
    rowIDList[next[v]++] = u;
  }
}

void runPR(uint64_t numVertices, std::vector<uint> &rowIDList, std::vector<uint> &colIDList, std::vector<uint64_t> &outDeg)
{
  PimDeviceProperties deviceProps;
  PimStatus status = pimGetDeviceProperties(&deviceProps);
  if (status != PIM_OK)
  {
    std::cout << "Abort" << std::endl;
    return;
  }

  uint64_t pimObjLength = numVertices;

  PimObjId rankObj = pimAlloc(PIM_ALLOC_AUTO, pimObjLength, PIM_UINT64);
  if (rankObj == -1)  {
    std::cout << "Abort" << std::endl;
    return;
  }

  uint64_t initRank = (1ULL << 24) / numVertices; // using fixed point representation with scaling factor of 2^16 to represent rank values between 0 and 1
  vector<uint64_t> initRankVec(pimObjLength, initRank), newRankVec(pimObjLength, 0);

  PimObjId outDegObj = pimAllocAssociated(rankObj, PIM_UINT64);
  if (outDegObj == -1)  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimCopyHostToDevice((void *)outDeg.data(), outDegObj);
  if (status != PIM_OK) {
    std::cout << "Abort copying outDeg to PIM device" << std::endl;
    return;
  }

  PimObjId danglingObj = pimAllocAssociated(rankObj, PIM_UINT64);
  if (danglingObj == -1)  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId condObj = pimAllocAssociated(rankObj, PIM_BOOL);
  if (condObj == -1)  {
    std::cout << "Abort" << std::endl;
    return;
  }

  PimObjId sumObj = pimAllocAssociated(rankObj, PIM_UINT64);
  if (sumObj == -1)  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimCopyHostToDevice((void *)initRankVec.data(), rankObj);
  if (status != PIM_OK)  {
    std::cout << "Aborting copying initRankVec to rankObj" << std::endl;
    return;
  }

  std::vector<uint8_t> nnzNeighborMaskVec(pimObjLength, 0);
  double eps = 1e-6;
  uint64_t eps_fp = (uint64_t)(eps * (1ULL << 24) + 0.5); // fixed point representation of epsilon

  PimObjId nonZeroNeighborObj = pimAllocAssociated(rankObj, PIM_BOOL);
  if (nonZeroNeighborObj == -1)  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimCopyHostToDevice((void *)nnzNeighborMaskVec.data(), nonZeroNeighborObj);
  if (status != PIM_OK)  {
    std::cout << "Abort copying non-zero neighbor mask to device" << std::endl;
    return;
  }

  PimObjId neighborRankObj = pimAllocAssociated(rankObj, PIM_UINT64);
  if (neighborRankObj == -1)  {
    std::cout << "Abort" << std::endl;
    return;
  }

  status = pimGTScalar(outDegObj, nonZeroNeighborObj, 0);
  if (status != PIM_OK)  {
    std::cout << "Abort computing non-zero neighbor mask" << std::endl;
    return;
  }

  PimObjId neighborDegObj = pimAllocAssociated(rankObj, PIM_UINT64);
  if (neighborDegObj == -1)  {
    std::cout << "Abort" << std::endl;
    return;
  }

  while (true) {
    uint64_t danglingSum = 0;

    status = pimBroadcastUInt(danglingObj, 0);
    if (status != PIM_OK)  {
      std::cout << "Abort broadcasting dangling sum" << std::endl;
      return;
    }

    status = pimEQScalar(outDegObj, condObj, 0);
    if (status != PIM_OK)  {
      std::cout << "Abort computing dangling mask" << std::endl;
      return;
    }

    status = pimCondCopy(condObj, rankObj, danglingObj);
    if (status != PIM_OK)  {
      std::cout << "Abort copying dangling ranks to dangling sum" << std::endl;
      return;
    }

    status = pimRedSum(danglingObj, static_cast<void*>(&danglingSum));
    if (status != PIM_OK)  {
      std::cout << "Abort computing dangling sum" << std::endl;
      return;
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    uint64_t danglingTerm = danglingSum / numVertices;
    uint64_t S = (1ULL << 24); // scaling factor for fixed point representation
    uint64_t F = 24; // number of fractional bits in fixed point representation
    uint64_t D = (uint64_t)(0.85 * (double)S + 0.5);
    uint64_t base = (S - D) / numVertices;
    auto end_cpu = std::chrono::high_resolution_clock::now();
    hostElapsedTime += end_cpu - start_cpu;

    for (uint64_t v = 0; v < numVertices; ++v) {
      auto start_cpu = std::chrono::high_resolution_clock::now();
      std::vector<uint8_t> neighborMaskVec(pimObjLength, 0);
      for (uint64_t i = colIDList[v]; i < colIDList[v + 1]; ++i) {
        neighborMaskVec[rowIDList[i]] = 1;
      }
      auto end_cpu = std::chrono::high_resolution_clock::now();
      hostElapsedTime += end_cpu - start_cpu;

      status = pimCopyHostToDevice(neighborMaskVec.data(), condObj);
      if (status != PIM_OK)  {
        std::cout << "Abort copying neighbor mask to device" << std::endl;
        return;
      }

      status = pimAnd(condObj, nonZeroNeighborObj, condObj);
      if (status != PIM_OK)  {
        std::cout << "Abort computing final neighbor mask" << std::endl;
        return;
      }

      status = pimBroadcastUInt(neighborRankObj, 0);
      if (status != PIM_OK)  {
        std::cout << "Abort broadcasting neighborRankObj" << std::endl;
        return;
      }

      status = pimCondCopy(condObj, rankObj, neighborRankObj);
      if (status != PIM_OK)  {
        std::cout << "Abort copying neighbor ranks to neighborRankObj" << std::endl;
        return;
      }

      status = pimBroadcastUInt(neighborDegObj, 1);
      if (status != PIM_OK)  {
        std::cout << "Abort broadcasting neighborDegObj" << std::endl;
        return;
      }

      status = pimCondCopy(condObj, outDegObj, neighborDegObj);
      if (status != PIM_OK)  {
        std::cout << "Abort copying neighbor degrees to neighborDegObj" << std::endl;
        return;
      }

      status = pimDiv(neighborRankObj, neighborDegObj, sumObj);
      if (status != PIM_OK)  {
        std::cout << "Abort dividing neighbor ranks by out degree" << std::endl;
        return;
      }

      uint64_t currSum = 0;
      status = pimRedSum(sumObj, static_cast<void*>(&currSum));
      if (status != PIM_OK)  {
        std::cout << "Abort computing sum of neighbor contributions" << std::endl;
        return;
      }

      start_cpu = std::chrono::high_resolution_clock::now();
      uint64_t acc = currSum + danglingTerm;   // all fixed-point
      uint64_t scaled = (D * acc) >> F;        // divide by S
      newRankVec[v] = base + scaled;
      end_cpu = std::chrono::high_resolution_clock::now();
      hostElapsedTime += end_cpu - start_cpu;
    }
    uint64_t maxDiff = 0;
    for (uint64_t v = 0; v < numVertices; ++v) {
      uint64_t diff = (newRankVec[v] > initRankVec[v]) ? (newRankVec[v] - initRankVec[v]) : (initRankVec[v] - newRankVec[v]);
      if (diff > maxDiff) maxDiff = diff;
    }
    if (maxDiff <= eps_fp) break;
    status = pimCopyHostToDevice((void *)newRankVec.data(), rankObj);
    if (status != PIM_OK)  {
      std::cout << "Aborting copying newRankVec to rankObj" << std::endl;
      return;
    }
    initRankVec.swap(newRankVec);
    std::cout << "Current max diff: " << (double)maxDiff / S << "\n";
  }

  // std::cout << "Dangling sum computed by PIM: " << (double)danglingSum << ", dangling sum computed by host: " << (double)hostDanglingSum << ", hostSum: " << (double)hostSum << "\n";
  pimFree(nonZeroNeighborObj);
  pimFree(neighborRankObj);
  pimFree(rankObj);
  pimFree(outDegObj);
  pimFree(danglingObj);
  pimFree(condObj);
  pimFree(sumObj);
  pimFree(neighborDegObj);
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
  std::vector<uint64_t> outDeg; // out degree of each vertex, needed for page rank computation
  // Read the edge list from the input file and construct CSR representation of the graph, keep track of the #vertices and #edges in the graph
  createCSC(fin, outDeg, rowIDList, colIDList, numVertices, numEdges);
  fin.close();
  std::cout << "Created CSC. Graph has " << numVertices << " vertices and " << numEdges << " edges.\n";

  if (!createDevice(params.configFile)) return 1;

  runPR(numVertices, rowIDList, colIDList, outDeg);
  if (params.shouldVerify) {
    // verify result
  }
  pimShowStats();
  std::cout << "\nHost elapsed time: " << hostElapsedTime.count() << " ms\n";

  return 0;
}
