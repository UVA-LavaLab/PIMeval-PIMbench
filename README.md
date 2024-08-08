## PIMeval Simulator and PIMbench Suite

![Contribution](https://img.shields.io/badge/Contribution-Welcome-blue) ![License](https://img.shields.io/badge/license-MIT-green.svg)

### Description
* PIMeval
  * A PIM simulation and evaluation framework implemented as a C++ library
  * Support various subarray-level bit-serial, subarray-level bit-parallel and bank-level PIM architectures
  * Support both vertical and horizontal data layouts
  * Support multi-PIM-core programming model and resource management
  * Support high-level functional simulation with a set of PIM architecture independent APIs
  * Support low-level micro-ops programming for modeling architecture details
  * Support performance and energy modeling with detailed stats tracking
  * Support multi-threaded simulation for runtime
* PIMbench
  * A rich set of PIM benchmark applications on top of the PIMeval functional simulation and evaluation framework

### Quick Start
```bash
git clone <url_to_your_fork>
cd PIMeval/
make -j<n_proc>
cd /PIMbench/<application_dir>/PIM
./<application_executable_name>.out
```

### Code Structure
* PIMeval: PIM similation framework - libpimeval
  * `libpimeval/src`: PIMeval simulator source code
  * `libpimeval.h`: PIMeval simulator library interface
  * `libpimeval.a`: PIMeval simulator library (after make)
* PIMbench: PIM benchmark suite
  * `cpp-aes`: AES encryption/decryption
  * `cpp-axpy`: aX+Y operation
  * `cpp-filter-by-key`: Filer by key
  * `cpp-gemm`: General matrix-matrix product
  * `cpp-gemv`: General matrix-vector product
  * `cpp-histogram`: Histogram
  * `cpp-brightness`: Image brightness
  * `cpp-image-downsampling`: Image downsampling
  * `cpp-kmeans`: K-means
  * `cpp-knn`: kNN
  * `cpp-linear-regression`: Linear regression
  * `cpp-radix-sort`: Radix sort
  * `cpp-triangle-count`: Triangle counting
  * `cpp-vec-add`: Vector addition
  * `cpp-convolution`: Convolution kernel for CNNs
  * `cpp-pooling`: Max pooling
  * `cpp-relu`: ReLU
  * `cpp-vgg13`: VGG-13
  * `cpp-vgg16`: VGG-16
  * `cpp-vgg19`: VGG-19
* misc-bench: Miscellenous benchmarks implemented for testing purpose.
  * `cpp-dot-prod`: Dot product
  * `cpp-prefixsum`: Prefix sum of a vector
  * `cpp-sad`: Sum of absolute difference
  * `cpp-vec-arithmetic`: Vector arithmetic
  * `cpp-vec-comp`: Vector comparison
  * `cpp-vec-div`: Vector division
  * `cpp-vec-logical`: Vector logical operations
  * `cpp-vec-mul`: Vector multiplication
  * `cpp-vec-popcount`: Vector popcount
  * `cpp-vec-broadcast-popcnt`: Vector broadcast and pop count
* Bit-serial micro-program evaluation framework
  * `bit-serial`
* Functional tests
  * `tests`

### How To Build
* Run `make` at root directory or subdirectories
  * `make perf`: Build with `-Ofast` for performance measurement (default)
  * `make debug`: Build with `-g` and `-DDEBUG` for debugging and printing verbose messages
<!--
  * `make dramsim3_integ`: Enable DRAMsim3 related code with `-DDRAMSIM3_INTEG`
-->
* Multi-threaded building
  * `make -j<n_proc>`
* Specify simulation target
  * `make PIM_SIM_TARGET=PIM_DEVICE_BITSIMD_V` (default)
  * `make PIM_SIM_TARGET=PIM_DEVICE_FULCRUM`
  * `make PIM_SIM_TARGET=PIM_DEVICE_BANK_LEVEL`
* Build with OpenMP
  * `make USE_OPENMP=1`
  * Guard any `-fopenmp` with this flag in Makefile used by a few applications

<!--
### About DRAMsim3 Integration
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
-->

### Contributors
This repository is the result of a collaborative effort by many individuals, including Farzana Ahmed Siddique, Deyuan Guo, Zhenxing Fan, Mohammadhosein Gholamrezaei, Morteza Baradaran, Alif Ahmed, Hugo Abbot, Kyle Durrer, Ethan Ermovick, Kumaresh Nandagopal and Khyati Kiyawat. We are grateful to everyone who contributed to this repository.

Special thanks go to Deyuan Guo for initially architecting the PIMeval simulator framework and bit-serial evaluation, and to Farzana Ahmed Siddique for her exceptional contributions to both the simulator and the PIMbench suite.

\<citation recommendation to be updated\>
### Contact

* Deyuan Guo - dg7vp AT virginia DOT edu
* Farzana Ahmed Siddique - farzana AT virginia DOT edu
* Kevin Skadron - skadron AT virginia DOT edu
