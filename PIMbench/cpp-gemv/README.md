# General Matrix Vector Multiplication (GEMV)

The GEMV is a Basic Linear Algebra Subprograms (BLAS) routine that performs a matrix-vector multiplication followed by a vector addition. The operation is defined as:

$y \leftarrow \alpha A x + \beta y$

where:
-  $\alpha$ and $\beta$ are scalars.
- $A$ is a matrix.
- $x$ and $y$ are vectors.

For a detailed description of GEMV, you can refer to the [BLAS GEMV documentation](http://www.netlib.org/blas/).

## Directory Structure

```
cpp-gemv/
├── PIM/
│   ├── Makefile
│   ├── gemv.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── gemv.cpp
│   ├── GPU/
│   │   ├── Makefile
│   │   ├── gemv.cu
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the GEMV benchmark:

1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The CPU variant of GEMV has been implemented using standard C++ library and OpenBLAS for parallel execution.

#### GPU

The GPU variant leverages CUDA C++ Core Libaries (CCCL) to perform GEMV on NVIDIA GPU.

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
./gemv.out
```

To see help text on all usages and how to modify any of the input parameters, use following command:

```bash
./gemv.out -h
```
