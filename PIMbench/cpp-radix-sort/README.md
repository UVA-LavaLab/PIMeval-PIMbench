# Radix Sort

Radix sort performs a sorting on an input array and outputs a sorted array.

## Directory Structure
```
cpp-radix-sort/
├── PIM/
│   ├── Makefile
│   ├── radix-sort.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── radix_sort.cpp
│   ├── GPU/
│   │   ├── include/
│   │   │   ├── bfloat16.h
│   │   │   ├── half.h
│   │   │   ├── mersenne.h
│   │   │   ├── test_util_vec.h
│   │   │   ├── test_util.h
│   │   ├── Makefile
│   │   ├── radix_sort.cu
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the radix sort benchmark:
1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The counting phase of radix sort has been implemented using standard C++ and OpenMP for parallel execution.

#### GPU

The GPU variant is INVIDIA's device-wide version of the radix sort from CUB library

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
Note that the GPU Makefile currently uses `SM_80`, which is compatible with the A100. To run it on a different GPU, please manually change this in the makefile.

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
./radix-sort.out
```

### For CPU version
```bash
./radix_sort.out <array_size>
```

### For GPU version

You can specify the input size using the `--n` option:

```bash
./radix_sort --n=65536
```
