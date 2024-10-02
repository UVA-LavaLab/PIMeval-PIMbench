# Linear Regression

Linear regression is an analytical and statisical model used to estimate the value of an input variable given the values of previously given data. For this use case, we are exploring simple linear regression, meaning there is only one explanatory variable.

## Directory Structure

```
cpp-linear-regression/
├── PIM/
│   ├── Makefile
│   ├── lr.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── lr.cpp
│   ├── GPU/
│   │   ├── Makefile
│   │   ├── lr.cu
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the linear regression benchmark:

1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The CPU variant of linear regression has been implemented using standard C++ library and OpenMP for parallel execution.

#### GPU

The GPU variant leverages CUDA C++ Core Libaries (CCCL) to perform linear regression on NVIDIA GPU.

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

*Note that the GPU Makefile currently uses SM_80, which is compatible with the A100. To run it on a different GPU, please manually change this in the makefile.

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
./lr.out
```

To see help text on all usages and how to modify any of the input parameters, use following command:

```bash
./lr.out -h
```
