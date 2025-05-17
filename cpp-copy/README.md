# Copy

The copy operation clones the data from one vector into another, and is described by the following:

$Y[i] \leftarrow X[i]$

where:
- $Y$ and $X$ are matrices.

## Directory Structure

```
cpp-copy/
├── PIM/
│   ├── Makefile
│   ├── copy.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── copy.cpp
│   ├── GPU/
│   │   ├── Makefile
│   │   ├── copy.cu
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the copy benchmark:

1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The CPU variant of copy has been implemented using the standard C++ library, as well as with parallelization from OpenMP.

#### GPU

The GPU variant leverages CUDA C++ to scale the vector on an NVIDIA GPU.

### PIM Implementation

The PIM variant is implemented using C++ with some speedup from OpenMP. Three different PIM architectures can be tested with this.

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
./copy.out
```

To see help text on all usages and how to modify any of the input parameters, use following command:

```bash
./copy.out -h
```
