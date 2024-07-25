# Vector Addition (VA)

Vector addition is a fundamental kernel in Linear Algebra that can be expressed as following equation:

$C[i] \leftarrow A[i] + B[i]$

where:
- $A$ and $B$ are input vectors.
- $C$ is the output vector.

## Directory Structure
```
cpp-vec-add/
├── PIM/
│   ├── Makefile
│   ├── vec-add.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── vec_add.cpp
│   ├── GPU/
│   │   ├── Makefile
│   │   ├── vec_add.cu
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the vector addition benchmark:
1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU has been used as baselines.

#### CPU

The CPU variant of vector addition has been implemented using standard C++ and OpenMP for parallel execution. The reason OpenBLAS was not used is OpenBLAS does not support $int$ datatype and PIMeval currently does not support $float$ or $double$

#### GPU

The GPU variant has been implemented using CUDA to perform element-wise addition of two vectors on NVIDIA GPU.

### PIM Implementation

The PIM variant is implemented using C++ and PIM architecture to perform element-wise addition of two vectors directly within the memory. The operation is defined as:

For a detailed description of vector addition using PIM, you can refer to the specific PIM architecture documentation.
  
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
*Note that the GPU Makefile currently uses `SM_80`, which is compatible with the A100. To run it on a different GPU, please manually change this in the makefile.

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
./vec_add.out
```

To see help text use following command
```bash
./vec_add.out
```

### Specifying Input Size

You can specify the input size using the `-` option:

```bash
./vec_add.out -l <input_size>
```

### Help and Usage Options

For help or to see usage options, use the `-h` option:

```bash
./vec_add.out -h
```
