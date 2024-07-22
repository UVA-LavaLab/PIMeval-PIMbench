## PIMeval Simulator

### Description

* This is a PIM functional simulator that can simulate various PIM devices, cores, resource allocation, and command execution
* This provides a C/C++ compatible library for PIM application development

### How To Build
* Run `make` at `pim-func-sim` root directory or subdirectories. Support three targets:
  * `make perf`: Build with `-O3` for performance measurement
  * `make debug`: Build with `-g` and `-DDEBUG` for debugging and printing verbose messages
  * `make dramsim3_integ`: Enable DRAMsim3 related code with `-DDRAMSIM3_INTEG`
* Multi-threaded building
  * `make -j10`
* Specify simulation target
  * `make PIM_SIM_TARGET=PIM_DEVICE_FULCRUM`
  * `make PIM_SIM_TARGET=PIM_DEVICE_BANK_LEVEL`
* Build with OpenMP
  * `make USE_OPENMP=1`
  * Guard any `-fopenmp` with this flag in Makefile

### Code Architecture
* libpimeval - PIMeval similation framework
  * `src`: PIMeval simualtor source code
  * `include/libpimeval.h`: Public header file (after make)
  * `lib/libpimeval.a`: PIMeval simulator library (after make)
* apps - PIMbench
  * `cpp-vec-add`: Vector addition
  * `cpp-vec-arithmetic`: Vector arithmetic
  * `cpp-vec-comp`: Vector comparison
  * `cpp-vec-logical`: Vector logical operations
  * `cpp-vec-broadcast-popcnt`: Vector broadcast and pop count
  * `cpp-gemv`: General matric-vector product
  * `cpp-sad`: Sum of absolute difference
  * `cpp-pooling`: Max pooling
  * `cpp-convolution`: Convolution
  * `cpp-linear-regression`: Linear regression
  * `cpp-aes`: AES encryption
  * `cpp-radix-sort`: Radix sort

### About DRAMsim3 Integration
* This module contains a copy of DRAMsim3
  * Oringal DRAMsim3 repo: https://github.com/umd-memsys/DRAMsim3
  * Clone date: 05/06/2024
  * Location: ./third_party/DRAMsim3/
* DRAMsim3 related code are guarded with DRAMSIM3_INTEG flag
  * Requires `make dramsim3_integ`
* Below is needed for dramsim3_integ for now
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
git clone <url_to_this_repo>
cd pim-func-sim
export DRAMSIM3_PATH=<path_to_DRAMSIM3>
make -j
```

### Contributors
* Deyuan Guo
* Farzana Siddique
* Mohammadhosein Gholamrezaei
* Zhenxing Fan
