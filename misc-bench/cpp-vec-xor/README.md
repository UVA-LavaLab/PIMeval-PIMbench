# XOR

XOR is a basic bitwise operation, which compares every bit place for two different vectors, and is popularly used in bitmap indexing and bitmap-based graph processing. If the bits are not equal at position `i`, then the resulting bit at position `i` is 1, else it is 0. The inclusion of this benchmark was inspired by previous works, such as the Fulcrum paper [[1]](#1).

For a more detailed description on the operation/algorithm, refer to the Microsoft XOR [documentation](https://learn.microsoft.com/en-us/dotnet/visual-basic/language-reference/operators/xor-operator).

## Directory Structure

```
cpp-vec-xor/
├── PIM/
│   ├── Makefile
│   ├── xor.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── xor.cpp
│   ├── GPU/
│   │   ├── Makefile
│   │   ├── xor.cu
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the XOR benchmark:

1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The CPU variant of histogram has been implemented using standard C++ library and OpenMP for parallel execution. This method was chosen as it was seen to be both the easiest and most effective in parallelizing the independent execution operations. 

#### GPU

The GPU variant leverages CUDA C++ Core Libaries (CCCL) to perform the XOR operation on NVIDIA GPU.

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

After compiling, run each executable with the following command that will run it with default parameters:

```bash
./xor.out
```

To see help text on all usages, use following command:

```bash
./xor.out -h
```

## References

<a id = "1">1.</a>
M. Lenjani et al., "Fulcrum: A Simplified Control and Access Mechanism Toward Flexible and Practical In-Situ Accelerators," 2020 IEEE International Symposium on High Performance Computer Architecture (HPCA), San Diego, CA, USA, 2020, pp. 556-569, doi: 10.1109/HPCA47549.2020.00052
