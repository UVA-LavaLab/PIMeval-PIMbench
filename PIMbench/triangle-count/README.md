# Triangle Counting (TC)

Triangle counting is a fundamental kernel in graph processing that counts the number of triangles in a graph. It is widely used in various applications, including social network analysis, fraud detection, and network security.

## Directory Structure
```
cpp-triangle-count/
├── Dataset/
├── PIM/
│   ├── Makefile
│   ├── tc.cpp
├── README.md
├── Makefile
```

## Implementation Description

This repository contains the implementation of triangle counting algorithm for PIM architectures.

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

For comparing to CPU, we used the GAP impelementation of triangle counting: https://github.com/sbeamer/gapbs.git

#### GPU

For comparing to GPU, we used the Gunrock impelementation of triangle counting: https://github.com/gunrock/gunrock.git

### PIM Implementation

The PIM variant is implemented using C++ and three different PIM architectures can be tested with this.
  
## Compilation Instructions for Specific Variants


### PIM Variant

To compile for the PIM variant, use:

```bash
cd PIM
make
```

## Execution Instructions

### Running the Executable

After compiling, run the executable with the following command that will run it for default parameters:

```bash
./tc.out
```

### Specifying Input file

You can specify the input file (provided in Dataset directory) using the `-i` option:

```bash
./tc.out -i <input_file>
```

### Help and Usage Options

For help or to see usage options, use the `-h` option:

```bash
./tc.out -h
```
