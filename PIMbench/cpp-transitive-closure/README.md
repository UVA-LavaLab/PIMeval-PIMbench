# Transitive Closure

Transitive closure is a graph analysis tool for directed and weighted graphs. This PIM implementation, along with the CPU and GPU benchmarks, uses the Floyd-Warhsall algorithm to find the shortest paths between all pairs of vertices. The incorporation of this benchmark into PIMbench was inspired from previous published works [[1]](#1). 

## Input Files

Currently, only specifically formatted .csv files can be used to test the functionality of this benchmark. The first line must contain the total number of nodes, then followed by the adjacency matrix. Within this adjacency matrix, all non-existent edges are represented by the value `inf`, which is then parsed as `MAX_EDGE_VALUE` in `transitive-closure.cpp`. The value of this macro can be changed to be greater or less depending on the requirements of your computation. Furthermore, the diagonal of the matrix should only contain `0`, as it is assumed there are no edges from a node to itself. Sample inputs can be found in the `/datafiles/` directory. Additional files that exceeded the file size for GitHub which were used in benchmarks for the paper can be found in the following Google Drive [folder](https://drive.google.com/drive/folders/1u6bKYfWPLlb-pL21hmCpmvXqPoRrJ3bN).

## Compilation Instructions

To compile, simply run:

```bash
$ make
```

In order to create a faster runtime with the algorithm, specifically with the `loadVector()` function, use:

```bash
$ make USE_OPENMP=1
```

## Execution Instructions

After compiling, run the executable using the following command:

```bash
$ ./transitive-closure.out
```

By default, a new adjacency matrix is generated with `numVertices` set at `256` with a `sparsityRate` of `50%`; however, you can specify these parameters with the `-l` and `-r` flags respectively:

```bash
$ ./transitive-closure.out -l <num_vertices>
```

or, 

```bash
$ ./transitive-closure.out -r <sparsity_rate>
```

If answer verification with a baseline CPU run is desired, use the `-v` flag:

```bash
$ ./transitive-closure.out -v t
```

The cases described above can all be used simultaneously. Additionally, you can specify a valid .csv file using the `-i` flag:

```bash
$ ./transitive-closure.out -i <input_file>
```

To see more details regarding the usages described above and more, use the `-h` option:

```bash
$ ./transitive-closure.out -h
```

## References

<a id = "1">[1]</a>
B. R. Gaeke, P. Husbands, X. S. Li, L. Oliker, K. A. Yelick and R. Biswas, "Memory-intensive benchmarks: IRAM vs. cache-based machines," Proceedings 16th International Parallel and Distributed Processing Symposium, Ft. Lauderdale, FL, USA, 2002, pp. 7 pp-, doi: 10.1109/IPDPS.2002.1015506.
