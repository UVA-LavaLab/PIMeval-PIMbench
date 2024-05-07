## PIM Functional Simulator

### Description

* This is a PIM functional simulator that can simulate various PIM devices, cores, resource allocation, and command execution
* This provides a C/C++ compatible library for PIM application development

### How To Build
* Run `make` to build the library and apps
* Below is temporarily disabled
```bash
# Build dramsim3
git clone https://github.com/fasiddique/DRAMsim3.git
cd DRAMsim3/
git checkout benchmark
mkdir build
cd build
cmake ..
make -j
# Build PIM functional simulator
git clone <url_to_your_fork>
cd dram-bitsimd-dev/pim-func-sim
export DRAMSIM3_PATH=<path_to_DRAMSIM3>
make -j
``` 

### Code Architecture
* libpimsim
  * `src`: PIM functional simualtor source code
  * `include/libpimsim.h`: Public header file (after make)
  * `lib/libpimsim.a`: PIM functional simulator library (after make)
* apps
  * `c-vec-add`: Simple vector addition demo app in C language
  * `cpp-vec-add`: Simple vector addition demo app in C++ language

### About DRAMsim3 Integration
* This module contains a copy of DRAMsim3
  * Oringal DRAMsim3 repo: https://github.com/umd-memsys/DRAMsim3
  * Clone date: 05/06/2024
  * Location: ./libpimsim/DRAMsim3/

### Contributors
* Deyuan Guo
* Farzana Siddique

