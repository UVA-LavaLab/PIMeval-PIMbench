# Random Forest (RF)

The random forest is a supervised model that utilizes an ensemble of decision trees to classify input data. 

## Directory Structure
```
random-forest/
├── PIM/
│   ├── Makefile
│   ├── rf.cpp
├── baselines/
│   ├── benchmark_rf.py
│   ├── run_rivanna_GPU.sh
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the vector addition benchmark:
1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The CPU variant of random forest has been implemented using the standard sklearn `RandomForestClassifer` object.

#### GPU

The GPU variant has been implemented using NVIDIA's highly optimized [Forest Inference library](https://developer.nvidia.com/blog/sparse-forests-with-fil/) library.
This is effectivly a wrapper around sklearn, so for the training RF, it's effectivly the same for both GPU and CPU. 

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

### Running the PIM Executable

After compiling, run the each executable with the following command that will run it for default parameters:

```bash
./rf.out
```

### Running the CPU and GPU File

After going to the `baselines` directory, run the python file with the following command that will run it for default parameters:

```bash
python ./benchmark_rf.py -cuda -num_trees 1000 -dt_height 6 -input_dim 20
```

Where the `cuda` flag will utilize a GPU to evaluate the model.

### Specifying Parameters

You can also specify the number of decision trees using the `-n` or `-num_trees` option and the height of each tree using the `-m` or `-dt_height` option:

```bash
./rf.out -n <number of Decision trees> -m <tree depth/height>
```

To specifiy how many dimensions for the input, use the `-d` or `-input_dim` option:

```bash
./rf.out -d <input dimension> 
```

### Help and Usage Options

For help or to see usage options, use the `-h` option:

```bash
./rf.out -h
```
