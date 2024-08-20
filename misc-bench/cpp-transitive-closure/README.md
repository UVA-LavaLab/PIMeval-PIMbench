# Transitive Closure

Transitive closure is a graph analysis tool for directed and weighted graphs. This PIM implementation, along with the CPU and GPU benchmarks, uses the Floyd-Warhsall algorithm to find the shortest paths between all pairs of vertices. The incorporation of this benchmark into PIMbench was inspired from previous published works [[1]](#1). 

## Input Files

Currently, only specifically formatted .csv files can be used to test the functionality of this benchmark. The first line must contain the total number of nodes, then followed by the adjacency matrix. Within this adjacency matrix, all non-existent edges are represented by the value `inf`, which is then parsed as `MAX_EDGE_VALUE` in `transitive-closure.hpp`, a .csv reader helper file. The value of this macro can be changed to be greater or less depending on the requirements of your computation. Furthermore, the diagonal of the matrix should only contain `0`, as it is assumed there are no edges from a node to itself. Sample inputs can be found in the `/datafiles/` directory. Additional files that exceeded the file size for GitHub which were used in benchmarks for the paper can be found in the following Google Drive [folder](https://drive.google.com/drive/folders/1u6bKYfWPLlb-pL21hmCpmvXqPoRrJ3bN).

## Directory Structure

```
cpp-transitive-closure/
├── PIM/
│   ├── Makefile
│   ├── transitive-closure.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── transitive-closure.cpp
│   ├── GPU/
│   │   ├── Makefile
│   │   ├── transitive-closure.cu
├── datafiles/
│   ├── 5V_50SR.csv
│   ├── 256V_50SR.csv
│   ├── 1024V_50SR.csv
│   ├── 2048_50SR.csv
├── transitive-closure.hpp
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the brightness benchmark:

1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The CPU variant of histogram has been implemented using standard C++ library and OpenMP for parallel execution. This method was chosen as it was seen to be the most effective in parallelizing the arithmetic for larger datasets when compared to other libraries such as Boost. 

#### GPU

The GPU variant leverages CUDA C++ Core Libaries (CCCL), specifically Thrust, to perform transitive closure on NVIDIA GPU.

### PIM Implementation

The PIM variant is implemented using C++ with OpenMP for speedup and three different PIM architectures can be tested with this.

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

In order to create a faster runtime with the algorithm, specifically when loading the addition and key vectors, use:

```bash
make USE_OPENMP=1
```

## Execution Instructions

### Running the Executable

After compiling, run each executable with the following command that will run it with default parameters:

```bash
./transitive-closure.out
```

To see help text on all usages, use following command:

```bash
./transitive-closure.out -h
```

### Specifying Matrix Generation Parameters

By default, a generated matrix with `256` nodes with a sparsity rate of `50%` is used as the input; however, you can specify these parameters with appropraite flags. To change the number of vertices, use the `-l` flag:

```bash
./transitive-closure.out -l <num_nodes>
```

To change the sparsity rate, use the `-r` flag:

```bash
./transitive-closure.out -r <sparsity_rate>
```

### Specifying Input File

Additionally, you can specify a valid .csv file as input by using the `-i` flag:

```bash
./transitive-closure.out -i <input_file>
```

## References

<a id = "1">[1]</a>
B. R. Gaeke, P. Husbands, X. S. Li, L. Oliker, K. A. Yelick and R. Biswas, "Memory-intensive benchmarks: IRAM vs. cache-based machines," Proceedings 16th International Parallel and Distributed Processing Symposium, Ft. Lauderdale, FL, USA, 2002, pp. 7 pp-, doi: 10.1109/IPDPS.2002.1015506.
