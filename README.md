## PIMeval Simulator and PIMbench Suite

![Contribution](https://img.shields.io/badge/Contribution-Welcome-blue) ![License](https://img.shields.io/badge/license-MIT-green.svg)

This repository contains the PIMeval simulator and the PIMbench benchmark suite, designed to evaluate Processing-In-Memory (PIM) architectures. For more details, please explore our [wiki](https://github.com/UVA-LavaLab/PIMeval-PIMbench/wiki) page, which provides comprehensive information on the tools, setup instructions, usage guidelines, documentation, contribution guideline and more!

### Description
**PIMeval** is a C++ library-based simulation and evaluation framework for Processing-in-Memory (PIM) systems. It supports a wide range of PIM architectures, including subarray-level bit-serial, bit-parallel, and bank-level designs, along with both vertical and horizontal data layouts. PIMeval facilitates multi-PIM-core programming models, resource management, and offers high-level functional simulations through architecture-independent APIs. Additionally, it provides low-level micro-ops programming for detailed architectural modeling, comprehensive performance and energy modeling with detailed statistics tracking, and multi-threaded simulation for efficient runtime.

**PIMbench** is a rich suite of benchmark applications built on top of the PIMeval framework. It leverages PIMeval's functional simulation and evaluation capabilities to provide diverse and comprehensive benchmarks, enabling thorough assessment and analysis of various PIM architectures.

<!--
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
-->

### Quick Start
```bash
git clone <url_to_your_fork>
cd PIMeval-PIMbench/
make -j<n_proc>
cd /PIMbench/<application_dir>/PIM
./<application_executable_name>.out
```

### Code Structure

```
PIMeval-PIMbench/
├── libpimeval/                        # PIM simulation framework
│   ├── src/                           # PIMeval simulator source code
├── PIMbench/                          # PIM benchmark suite
│   ├── aes/                           # AES encryption/decryption
│   │   ├── PIM/                       # PIM implementation of AES
│   │   └── baselines/                 # Baseline implementations
│   │       ├── CPU/                    # CPU baseline for AES
│   │       └── GPU/                    # GPU baseline for AES
│   ├── axpy/                          # aX+Y operation
│   │   ├── PIM/                       # PIM implementation of aX+Y
│   │   └── baselines/                 # Baseline implementations
│   │       ├── CPU/                    # CPU baseline for aX+Y
│   │       └── GPU/                    # GPU baseline for aX+Y
│   ├── filter-by-key/                 # Filter by key
│   │   ├── PIM/                       # PIM implementation of Filter by Key
│   │   └── baselines/                 # Baseline implementations
│   │       ├── CPU/                    # CPU baseline for Filter by Key
│   │       └── GPU/                    # GPU baseline for Filter by Key
│   ├── gemm/                          # General matrix-matrix product
│   │   ├── PIM/                       # PIM implementation of GEMM
│   │   └── baselines/                 # Baseline implementations
│   │       ├── CPU/                    # CPU baseline for GEMM
│   │       └── GPU/                    # GPU baseline for GEMM
│   ├── gemv/                          # General matrix-vector product
│   │   ├── PIM/                       # PIM implementation of GEMV
│   │   └── baselines/                 # Baseline implementations
│   │       ├── CPU/                    # CPU baseline for GEMV
│   │       └── GPU/                    # GPU baseline for GEMV
│   ├── histogram/                     # Histogram
│   │   ├── PIM/                       # PIM implementation of Histogram
│   │   └── baselines/                 # Baseline implementations
│   │       ├── CPU/                    # CPU baseline for Histogram
│   │       └── GPU/                    # GPU baseline for Histogram
│   ├── brightness/                    # Image brightness
│   │   ├── PIM/                       # PIM implementation of Image Brightness
│   │   └── baselines/                 # Baseline implementations
│   │       ├── CPU/                    # CPU baseline for Image Brightness
│   │       └── GPU/                    # GPU baseline for Image Brightness
│   ├── image-downsampling/            # Image downsampling
│   │   ├── PIM/                       # PIM implementation of Image Downsampling
│   │   └── baselines/                 # Baseline implementations
│   │       ├── CPU/                    # CPU baseline for Image Downsampling
│   │       └── GPU/                    # GPU baseline for Image Downsampling
│   ├── kmeans/                        # K-means
│   │   ├── PIM/                       # PIM implementation of K-means
│   │   └── baselines/                 # Baseline implementations
│   │       ├── CPU/                    # CPU baseline for K-means
│   │       └── GPU/                    # GPU baseline for K-means
│   ├── knn/                           # kNN
│   │   ├── PIM/                       # PIM implementation of kNN
│   │   └── baselines/                 # Baseline implementations
│   │       ├── CPU/                    # CPU baseline for kNN
│   │       └── GPU/                    # GPU baseline for kNN
│   ├── linear-regression/             # Linear regression
│   │   ├── PIM/                       # PIM implementation of Linear Regression
│   │   └── baselines/                 # Baseline implementations
│   │       ├── CPU/                    # CPU baseline for Linear Regression
│   │       └── GPU/                    # GPU baseline for Linear Regression
│   ├── radix-sort/                    # Radix sort
│   │   ├── PIM/                       # PIM implementation of Radix Sort
│   │   └── baselines/                 # Baseline implementations
│   │       ├── CPU/                    # CPU baseline for Radix Sort
│   │       └── GPU/                    # GPU baseline for Radix Sort
│   ├── triangle-count/                # Triangle counting
│   │   ├── PIM/                       # PIM implementation of Triangle Counting
│   ├── vec-add/                       # Vector addition
│   │   ├── PIM/                       # PIM implementation of Vector Addition
│   │   └── baselines/                 # Baseline implementations
│   │       ├── CPU/                    # CPU baseline for Vector Addition
│   │       └── GPU/                    # GPU baseline for Vector Addition
│   ├── convolution/                   # Convolution kernel for CNNs
│   │   ├── PIM/                       # PIM implementation of Convolution
│   │   └── baselines/                 # Baseline implementations
│   │       ├── CPU/                    # CPU baseline for Convolution
│   │       └── GPU/                    # GPU baseline for Convolution
│   ├── pooling/                       # Max pooling
│   │   ├── PIM/                       # PIM implementation of Max Pooling
│   │   └── baselines/                 # Baseline implementations
│   │       ├── CPU/                    # CPU baseline for Max Pooling
│   │       └── GPU/                    # GPU baseline for Max Pooling
│   ├── relu/                          # ReLU
│   │   ├── PIM/                       # PIM implementation of ReLU
│   │   └── baselines/                 # Baseline implementations
│   │       ├── CPU/                    # CPU baseline for ReLU
│   │       └── GPU/                    # GPU baseline for ReLU
│   ├── vgg13/                         # VGG-13
│   │   ├── PIM/                       # PIM implementation of VGG-13
│   │   └── baselines/                 # Baseline implementations
│   ├── vgg16/                         # VGG-16
│   │   ├── PIM/                       # PIM implementation of VGG-16
│   │   └── baselines/                 # Baseline implementations
│   └── vgg19/                         # VGG-19
│       ├── PIM/                       # PIM implementation of VGG-19
│       └── baselines/                 # Baseline implementations
├── misc-bench/                        # Miscellaneous benchmarks for testing purposes
│   ├── cpp-dot-prod/                  # Dot product
│   ├── cpp-prefixsum/                 # Prefix sum of a vector
│   ├── cpp-sad/                       # Sum of absolute difference
│   ├── cpp-vec-arithmetic/            # Vector arithmetic
│   ├── cpp-vec-comp/                  # Vector comparison
│   ├── cpp-vec-div/                   # Vector division
│   ├── cpp-vec-logical/               # Vector logical operations
│   ├── cpp-vec-mul/                   # Vector multiplication
│   ├── cpp-vec-popcount/              # Vector popcount
│   └── cpp-vec-broadcast-popcnt/      # Vector broadcast and pop count
├── bit-serial/                        # Bit-serial micro-program evaluation framework
│   └── bit-serial/                    # [Additional contents if any]
└── tests/                             # Functional tests
    └── tests/                         # [Additional test files or directories]
```
<!--
### How To Build
* Run `make` at root directory or subdirectories
  * `make perf`: Build with `-Ofast` for performance measurement (default)
  * `make debug`: Build with `-g` and `-DDEBUG` for debugging and printing verbose messages
* Multi-threaded building
  * `make -j<n_proc>`
* Specify simulation target
  * `make PIM_SIM_TARGET=PIM_DEVICE_BITSIMD_V` (default)
  * `make PIM_SIM_TARGET=PIM_DEVICE_FULCRUM`
  * `make PIM_SIM_TARGET=PIM_DEVICE_BANK_LEVEL`
* Build with OpenMP
  * `make USE_OPENMP=1`
  * Guard any `-fopenmp` with this flag in Makefile used by a few applications
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

Special thanks go to Deyuan Guo for initially architecting the PIMeval simulator framework and bit-serial evaluation, and to Farzana Ahmed Siddique for her contributions to both the simulator and the PIMbench suite.

\<citation recommendation to be updated\>
### Contact

* Deyuan Guo - dg7vp AT virginia DOT edu
* Farzana Ahmed Siddique - farzana AT virginia DOT edu
* Kevin Skadron - skadron AT virginia DOT edu
