# Radix Sort

'Filter by key' application is a typical application seen in DBMS where a certain (or a group of) critaria(s) were applied to a data set and only the data that satisfies the critaria are 'filtered' as the output, in this implementation, we simply apply a 'smaller than' comparison between each element of an int vector and an int 'key'.

## Directory Structure
```
cpp-filter-by-key/
├── PIM/
│   ├── Makefile
│   ├── db-filtering.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── filter.cpp
│   ├── GPU/
│   │   ├── include/
│   │   │   ├── bfloat16.h
│   │   │   ├── half.h
│   │   │   ├── mersenne.h
│   │   │   ├── test_util_vec.h
│   │   │   ├── test_util.h
│   │   ├── Makefile
│   │   ├── filtering.cu
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

The filtering step has been implemented using standard C++ and OpenMP for parallel execution.

#### GPU

The GPU variant is INVIDIA's device-wide version of the 'select' from CUB library

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
./filter.out -n 65536
-n    database size (default=65536 elements)"
-k    value of key (default = 1)"
-v    t = print output vector. (default=false)"
```

### For GPU version

You can specify the input size using the `--n` option:

```bash
./filtering --n=65536
```
