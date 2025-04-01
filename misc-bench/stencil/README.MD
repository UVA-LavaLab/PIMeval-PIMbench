# Stencil

Stencil computing takes a 2D array and sets each element to the average of its neighbors and itself. By default, the average is taken of each 3x3 block of elements, however this can be adjusted using the runtime parameters. For this benchmark, elements with neighbors outside of the input range are not computed. Additionally, the location of the output element within the stencil grid can be changed via the paramemters, meaning each output element could be the average of the 3x3 block to its bottom right etc., rather than being just in the center. For example, averaging a 3x3 grid with a 3x3 stencil pattern, with num left and num above both set to 1:

- Input: [[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
- Output: [[#, #, #], [#, 1.66, #], [#, #, #]]
- \# represents an undefined element

## Directory Structure

```
stencil/
├── PIM/
│   ├── Makefile
│   ├── stencil.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── stencil.cpp
│   ├── GPU/
│   │   ├── Makefile
│   │   ├── stencil.cu
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the stencil benchmark:

1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The CPU variant of stencil has been implemented using the standard C++ library, as well as with parallelization from OpenMP.

#### GPU

The GPU variant leverages CUDA C++ to parallelize the stencil benchmark on an NVIDIA GPU.

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
./stencil.out
```

To see help text on all usages and how to modify any of the input parameters, use the following command:

```bash
./stencil.out -h
```
