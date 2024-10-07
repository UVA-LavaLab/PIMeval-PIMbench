# Game of Life

The Game of Life is a zero player game, where a board of cells progress through various states of being alive or dead at each step. This program simulates a single state transition. The rules for state transitions are described below:

- Cells are live in the next state if either:
  - They are dead, and have exactly 3 live neighbors
  - They are alive, and have 2 or 3 live neighbors
- All other cells will be dead in the next iteration

Note that neighbors are defined as the block of 8 cells surrounding a cell, including diagnals. Additionally, neighbors that are off the edge of the board are treated as dead.

## Directory Structure

```
cpp-game-of-life/
├── PIM/
│   ├── Makefile
│   ├── game-of-life.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── game-of-life.cpp
│   ├── GPU/
│   │   ├── Makefile
│   │   ├── game-of-life.cu
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the game of life benchmark:

1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The CPU variant of game of life has been implemented using the standard C++ library, as well as with parallelization from OpenMP.

#### GPU

The GPU variant leverages CUDA C++ to parallelize game of life on an NVIDIA GPU.

### PIM Implementation

The PIM variant is implemented using C++ with some speedup from OpenMP. Three different PIM architectures can be tested with this.

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

After compiling, run the each executable with the following command that will run it for default parameters:

```bash
./game-of-life.out
```

To see help text on all usages and how to modify any of the input parameters, use following command:

```bash
./game-of-life.out -h
```
