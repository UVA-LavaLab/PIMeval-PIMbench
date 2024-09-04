# Scale

Scaling is an operation on a vector that is given by the following formula:

$Y[i] \leftarrow \alpha X[i]$

where:
-  $\alpha$ is a scalar.
- $Y$ and $X$ are matrices.

## Directory Structure

```
cpp-scale/
├── PIM/
│   ├── Makefile
│   ├── scale.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── scale.cpp
│   ├── GPU/
│   │   ├── Makefile
│   │   ├── scale.cu
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the scale benchmark:

1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The CPU variant of scale has been implemented using the standard C++ library.

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
./scale.out
```

To see help text on all usages and how to modify any of the input parameters, use following command:

```bash
./scale.out -h
```
