# Breadth First Search (BFS)

Breadth First Search (BFS) is a graph traversal algorithm, and the version that is implemented in PIMbench is the one wihich explores a graph **level-by-level** starting from a **source vertex** $s$. BFS visits all vertices reachable from $s$ using a FIFO queue.

## Directory Structure
```
bfs/
├── PIM/
│   ├── Makefile
│   ├── bfs.cpp
├── README.md
├── Makefile
```

## Implementation Description

This repository contains PIM implementations.

### Baseline Implementation

We use CPU and GPU implementations as baselines.

#### CPU

For the CPU baseline, we use the BFS implementation from the GAP Benchmark Suite \[[GitHub](https://github.com/sbeamer/gapbs)\].

#### GPU

For the GPU baseline, we use the BFS implementation from Gunrock \[[GitHub](https://github.com/gunrock/gunrock)\].

### PIM Implementation

The PIM variant is implemented using C++ and different PIM architectures can be tested with this.
  
## Compilation Instructions for Specific Variants

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
./bfs.out
```

To see help text on all usages, use following command
```bash
./bfs.out -h
```

### Specifying Input

You can specify the input file containing edge list using the `-i` option:

```bash
./vec-add.out -i <your_input.el>
```