# Bitmap

Bitmap indexing is a data indexing technique used to create massive speedup for database read-only querying. This algorithm creates a bitmap index, which is a string-like data structure storing the presence or absense of certain specified data values. The inclusion of this benchmark was inspired by previous works, such as the Fulcrum paper [[1]](#1), and closely models its PIM implementation by keeping the use of comparing and bitshifting operations consistent. The PIM, CPU, and GPU implementations are currently only statically built, meaning that the number of bitmap indices is constant and set to `8`. Further work should add functionality to control this parameter and dynamically change the data type stored in the PIM device.

For a more detailed description on the operation/algorithm, refer to the Oracle bitmap indexing [documentation](https://docs.oracle.com/cd/B10500_01/server.920/a96520/indexes.htm).

## Directory Structure

```
cpp-bitmap/
├── PIM/
│   ├── Makefile
│   ├── bitmap.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── bitmap.cpp
│   ├── GPU/
│   │   ├── Makefile
│   │   ├── bitmap.cu
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the bitmap benchmark:

1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The CPU variant of bitmap has been implemented using standard C++ library and OpenMP for parallel execution. This method was chosen as it was seen to be both the easiest and most effective in parallelizing the independent execution operations. 

#### GPU

The GPU variant leverages CUDA C++ Core Libaries (CCCL) to perform the bitmap indexing on NVIDIA GPU.

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
./bitmap.out
```

To see help text on all usages, use following command:

```bash
./bitmap.out -h
```

## References

<a id = "1">1.</a>
M. Lenjani et al., "Fulcrum: A Simplified Control and Access Mechanism Toward Flexible and Practical In-Situ Accelerators," 2020 IEEE International Symposium on High Performance Computer Architecture (HPCA), San Diego, CA, USA, 2020, pp. 556-569, doi: 10.1109/HPCA47549.2020.00052
