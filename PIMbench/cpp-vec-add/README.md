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
│   │   ├── vec-add-gpu.cu
├── README.md
```

## Implementation Description

This repository contains three different implementation of vector addition benchmark: 1. CPU baseline, 2. GPU baseline, 3. PIM

### Baseline Implementation

#### CPU

The CPU variant of vector addition has been implemented using standard C++ and OpenMP for parallel execution. The reason OpenBLAS was not used is OpenBLAS does not support $int$ datatype and PIMeval currently does not support $float$ or $double$

#### GPU

The GPU variant leverages cuBLAS to perform element-wise addition of two vectors on NVIDIA GPU.

### PIM Implementation

The PIM (Processing In Memory) variant utilizes PIM architecture to perform element-wise addition of two vectors directly within the memory. The operation is defined as:

For a detailed description of vector addition using PIM, you can refer to the specific PIM architecture documentation.
  
## Compilation Instructions for Specific Variants

### CPU Variant

To compile for the CPU variant, use:

```bash
make CPU
```

### GPU Variant

To compile for the GPU variant, use:

```bash
make GPU
```

### PIM Variant

To compile for the PIM variant, use:

```bash
make PIM
```

## Execution Instructions

### Running the Executable

After compiling, run the executable with the following command:

```bash
./vec_add.out
```

### Specifying Input Size

You can specify the input size using the `-i` option:

```bash
./vec_add.out -i <input_size>
```

### Help and Usage Options

For help or to see usage options, use the `-h` option:

```bash
./vec_add.out -h
```
