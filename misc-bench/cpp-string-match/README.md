# Scale

String match is an operation involving two strings that determines indices where one string, the needle, appears in another string, the haystack. This particular version of string match creates an array of the length haystack where every position indicates whether there is a match of the needle and the haystack starting at that position. For example:

string_match("abcdabc", "abc") -> [1, 0, 0, 0, 1, 0, 0]

## Directory Structure

```
cpp-string-match/
├── PIM/
│   ├── Makefile
│   ├── string-match.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── string-match.cpp
│   ├── GPU/
│   │   ├── Makefile
│   │   ├── string-match.cu
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the string match benchmark:

1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The CPU variant of string has been implemented using the standard C++ library, as well as with parallelization from OpenMP.

#### GPU

The GPU variant leverages CUDA C++ to find string matches on a GPU.

### PIM Implementation

The PIM variant is implemented using C++ with some speedup from OpenMP. Three different PIM architectures can be tested with this.

## Compilation Instructions for Specific Variants

### CPU Variant

To compile for the CPU variant, use:

```bash
cd baselines/CPU
apptainer build string-match.sif string-match-container.def
apptainer run string-match.sif
mkdir build
cd build
cmake ..
make
cd src
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
./string-match.out
```

To see help text on all usages and how to modify any of the input parameters, use following command:

```bash
./string-match.out -h
```
