# String Match

String match is an operation involving two strings that determines indices where one string, the needle, appears in another string, the haystack. This particular version of string match creates an array of the length of the haystack where every position indicates whether there is a match of the needle and the haystack starting at that position, for 1 or more needles and 1 haystack. For example:

string_match("abcdabc", ["abc"]) -> [1, 0, 0, 0, 1, 0, 0]

## Directory Structure

```
cpp-string-match/
├── PIM/
│   ├── Makefile
│   ├── string-match.cpp
├── baselines/
│   ├── CPU/
│   │   ├── CMakeLists.txt
│   │   ├── string-match-container.def
│   │   ├── .gitignore
│   │   ├── hyperscan/ (submodule)
│   │   ├── src/
│   │   │   ├── CMakeLists.txt
│   │   │   ├── string-match.cpp
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

The CPU variant of string matching uses [hyperscan](https://github.com/intel/hyperscan) to match the strings.

#### GPU

The GPU variant leverages CUDA C++ to find string matches on a GPU.

### PIM Implementation

The PIM variant is implemented using C++ with some speedup from OpenMP. Three different PIM architectures can be tested with this.

## Compilation Instructions for Specific Variants

### CPU Variant

The CPU varient requires [apptainer](https://apptainer.org/) to run the container to compile in (but can run outside of the container). To compile for the CPU variant, use:

```bash
cd baselines/CPU
apptainer build string-match.sif string-match-container.def
apptainer run string-match.sif
mkdir -p build
cd build
cmake ..
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

After compiling, run the each executable with the following command that will run it for default parameters (Must be run in the build directory for the CPU baseline):

```bash
./string-match.out
```

To see help text on all usages and how to modify any of the input parameters, use following command:

```bash
./string-match.out -h
```
