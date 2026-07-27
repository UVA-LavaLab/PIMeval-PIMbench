#include <stdio.h>
#include <cstdlib>
#include <getopt.h>
#include <assert.h>
#include <chrono>
#include <iomanip>
#include <algorithm>

#include "util.h"
#include "libpimeval.h"

typedef struct Params
{
  uint64_t numChasers;
  uint64_t tableSize;
  uint64_t numHops;
  char *configFile;
} Params;

void usage()
{
  fprintf(stderr,
    "\nUsage:  ./gather.out [options]"
    "\n"
    "\n    -n    number of chasers (start ptrs) (default=65536)"
    "\n    -t    table size (default=200000)"
    "\n    -i    number of hops (default=5)"
    "\n    -c    dramsim config file"
    "\n"
  );
}

struct Params input_params(int argc, char **argv)
{
  struct Params p;
  p.numChasers = 65536;
  p.tableSize = 200000;
  p.numHops = 5;
  p.configFile = nullptr;

  int opt;
  while ((opt = getopt(argc, argv, "hn:t:i:c:")) >= 0) {
    switch (opt)
    {
      case 'h':
        usage();
        exit(0);
        break;
      case 'n':
        p.numChasers = atoi(optarg);
        break;
      case 't':
        p.tableSize = atoi(optarg);
        break;
      case 'i':
        p.numHops = atoi(optarg);
        break;
      case 'c':
        p.configFile = optarg;
        break;
      default:
        fprintf(stderr, "\nUnrecognized option!\n");
        usage();
        exit(0);
    }
  }

  assert(p.numChasers >= 0 && "Invalid number of chasers!");
  assert(p.tableSize >= 0 && "Invalid table size!");
  assert(p.numChasers >= 0 && "Invalid number of hops!");
  return p;
}

int main(int argc,char **argv)
{
  struct Params p = input_params(argc, argv);

  if (!createDevice(p.configFile))
  {
    return 1;
  }

  std::vector<uint64_t> host_table(p.tableSize);
  std::vector<uint64_t> host_idx(p.numChasers);
  std::vector<uint64_t> host_dest(p.numChasers, 0); // fill with 0

  std::mt19937 gen{std::random_device{}()};
  std::uniform_int_distribution<uint64_t> dist(0, p.tableSize - 1);

  for (uint64_t i = 0; i < p.tableSize; i++) {
    host_table[i] = dist(gen);
  }
  
  std::vector<uint64_t> all_indices(p.tableSize);
  std::iota(all_indices.begin(), all_indices.end(), 0);
  std::shuffle(all_indices.begin(), all_indices.end(), gen);

  for (uint64_t i = 0; i < p.numChasers; i++) {
    host_idx[i] = all_indices[i];
  }

  int track_limit = std::max(1, std::min((int)p.numChasers, 5)); 
  uint64_t step = p.numChasers / track_limit;
  std::vector<std::vector<uint64_t>> chaser_paths(track_limit);
  for (int j = 0; j < track_limit; j++) {
    uint64_t target_idx = j * step;
    chaser_paths[j].push_back(host_idx[target_idx]); 
  }

  std::cout << "\n---------------------------------------------" << std::endl;
  std::cout << "--- MEMORY TABLE (Index -> Value) ---" << std::endl;
  int print_limit = std::min((int)p.tableSize, 20); // only print first 20 to avoid flooding terminal
  for(int i = 0; i < print_limit; i++) {
      std::cout << "[" << std::setw(2) << std::setfill('0') << i << "]->" 
                << std::setw(4) << std::setfill('0') << host_table[i] << "   ";
      if ((i + 1) % 5 == 0) std::cout << std::endl;
  }
  
  std::cout << "\n--- CHASER STARTING POSITIONS ---" << std::endl;
  int chaser_limit = std::min((int)p.numChasers, 5); // Only print 5 chasers
  for(int i = 0; i < chaser_limit; i++) {
    uint64_t target_idx = i * step;
    std::cout << "Chaser " << target_idx << " starts at index: " << host_idx[target_idx] << std::endl;
  }
  std::cout << "---------------------------------------------\n\n";

  PimObjId pim_table = pimAlloc(PIM_ALLOC_AUTO, p.tableSize, PIM_UINT64);
  assert(pim_table != -1);
  PimObjId pim_idx = pimAlloc(PIM_ALLOC_AUTO, p.numChasers, PIM_UINT64);
  assert(pim_idx != -1);
  PimObjId pim_dest = pimAlloc(PIM_ALLOC_AUTO, p.numChasers, PIM_UINT64);
  assert(pim_dest != -1);

  PimStatus status = pimCopyHostToDevice(host_table.data(), pim_table);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice(host_idx.data(), pim_idx);
  assert(status == PIM_OK);
  status = pimCopyHostToDevice(host_dest.data(), pim_dest);
  assert(status == PIM_OK);

  std::cout << "PIM Pointer Chase (" << p.numHops << " hops)" << std::endl;

  for (uint64_t i = 0; i < p.numHops; i++) {
    std::cout << "  -> Executing Hop " << i + 1 << std::endl;
    status = pimGather(pim_table, pim_idx, pim_dest);
    assert(status == PIM_OK);

    pimCopyDeviceToHost(pim_dest, host_dest.data()); 
    for (int j = 0; j < track_limit; j++) {
      uint64_t target_idx = j * step;
        chaser_paths[j].push_back(host_dest[target_idx]);
    }

    std::swap(pim_idx, pim_dest);
  }

  status = pimCopyDeviceToHost(pim_idx, host_dest.data()); // final result is in pim_idx instead of pim_dest
  assert(status == PIM_OK);

  std::cout << "\n-------------------------------------------" << std::endl;
  std::cout << "--- Pointer Chase Paths ---" << std::endl;
  for (int j = 0; j < track_limit; j++) {
      uint64_t target_idx = j * step;
      std::cout << "Chaser " << target_idx << ": ";
      for (size_t k = 0; k < chaser_paths[j].size(); k++) {
          std::cout << std::setw(4) << std::setfill('0') << chaser_paths[j][k];
          if (k < chaser_paths[j].size() - 1) std::cout << " -> ";
      }
      std::cout << std::endl;
  }
  std::cout << "-------------------------------------------\n" << std::endl;

  std::cout << "Results: " << std::endl;

  for (uint64_t i = 0; i < (uint64_t)std::min((int)p.numChasers, 5); i++) { // print first 5 results
    uint64_t target_idx = i * step;
    std::cout << "Chaser " << target_idx << " -> Fetched value: " << host_dest[target_idx] << std::endl;
  }

  pimFree(pim_table);
  pimFree(pim_idx);
  pimFree(pim_dest);

  pimShowStats();

  return 0;
}
