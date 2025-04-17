# K-Nearest Neighbors (KNN)

The KNN algorithm is a supervised learning classifier which uses proximity to classify the grouping of an individual query point. In our implementation of KNN, we use Manhatten Distance which is defined as:

$ d(x,y) = \sum_{i=1}^{m}|x_i - y_i|$

## Directory Structure
```
cpp-knn/
├── PIM/
│   ├── Makefile
│   ├── knn.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── knn.cpp
│   ├── GPU/
│   │   ├── inc/
│   │   ├── knncuda.cu
│   │   ├── knncuda.h
│   │   ├── Makefile
│   │   ├── test.cpp
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the vector addition benchmark:
1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The CPU variant of k nearest neighbors has been implemented using standard C++ and OpenMP for parallel execution.

#### GPU

The GPU variant has been implemented using CUDA to perform the KNN process end to end using global memory. Note that the inc directory was pulled from [this](https://github.com/MarziehLenjani/InSituBench/tree/master) repo.

### PIM Implementation

The PIM variant is implemented using C++ and three different PIM architectures can be tested with this.
  
## Compilation Instructions for Specific Variants

### CPU Variant

To compile for the CPU variant, use:

```bash
cd baselines/CPU
make
```

### GPU Variant

To compile for the GPU variant, use:

```bash
cd baselines/GPU
make
```

### PIM Variant

To compile for the PIM variant, use:

```bash
cd PIM
make
```

## Execution Instructions

### Running the Executable

After compiling, run the each executable with the following command that will run it for default parameters:

```bash
./knn.out
```

### Specifying Parameters

You can also specify the number of reference points using the -n option and the number of query points using the -m option:

```bash
./knn.out -n <reference_size> -m <query_size>
```

To specifiy how many k neighbors to consider, use the -k option:

```bash
./knn.out -k <number_of_neighbors> 
```

To specifiy the dimension of data points use the -k option:

```bash
./knn.out -d <dimension> 
```

### Help and Usage Options

For help or to see usage options, use the `-h` option:

```bash
./knn.out -h
```
