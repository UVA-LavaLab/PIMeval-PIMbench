# General Matrix Vector Multiplication - Batched (GEMV-Batched)

The GEMV-Batched is a Basic Linear Algebra Subprograms (BLAS) routine that extends the standard GEMV operation to handle multiple matrix-vector multiplications simultaneously. The operation is defined as:

$y_i \leftarrow \alpha A_i x_i + \beta y_i$

for $i = 1, 2, \dots, n$ where:
- $\alpha$ and $\beta$ are scalars.
- $A_i$ is the i-th matrix.
- $x_i$ and $y_i$ are the i-th vectors.
- $n$ is the number of batched operations.

For a detailed description of GEMV-Batched, you can refer to the [BLAS GEMV documentation](http://www.netlib.org/blas/).

## Directory Structure

```
cpp-gemv-batched/
├── baselines/
│   ├── CPU/
│   │   ├── gemv-batched.cpp
│   │   ├── gemv-batched.out
│   │   ├── Makefile
│   ├── GPU/
│   │   ├── gemv-batched.cu
│   │   ├── gemv-batched.out
│   │   ├── Makefile
├── PIM/
│   ├── gemv-batched.cpp
│   ├── gemv-batched.out
│   ├── Makefile
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the GEMV-Batched benchmark:

1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The CPU variant of GEMV-Batched has been implemented using standard C++ library and OpenBLAS for parallel execution across multiple GEMV operations.

#### GPU

The GPU variant leverages CUDA C++ Core Libraries (CCCL) to perform batched GEMV operations on NVIDIA GPU.

### PIM Implementation

The PIM variant is implemented using C++ and allows testing across different PIM architectures.

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

*Note that the GPU Makefile currently uses SM_80, which is compatible with the A100. To run it on a different GPU, please manually change this in the Makefile.

### PIM Variant

To compile for the PIM variant, use:

```bash
cd PIM
make
```

## Execution Instructions

### Running the Executable

After compiling, run each executable with the following command that will run it for default parameters:

```bash
./gemv-batched.out
```

To see help text on all usages and how to modify any of the input parameters, use the following command:

```bash
./gemv-batched.out -h
```

---

This README file is designed to match the style and structure of your original README while providing the necessary details for the GEMV-Batched kernel.
