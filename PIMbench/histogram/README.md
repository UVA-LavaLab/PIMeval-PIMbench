# Histogram

Histogram is a basic image processing algorithm that analyzes the number of instances that fall into a predetermined number of bins. For our use case, we analyze 24-bit .bmp files and create 256 bins for 3 separate channels. The number of bins is significant as they represent each key value from 0 to 255, while the channels corresponds to either red, green, or blue (RGB) color characteristics within the image. The .bmp file format is chosen as data is easy to extract and perform arithmetic on. Additionally, previous works, such as Phoenix [[1]](#1), use the same format and implementation. This PIM implementation uses sequential memory accessing, instead of the random accessing that is utilized in the CPU and GPU benchmarks.

## Input Files

As mentioned above, only 24-bit .bmp files can currently be used to test the functionality of this benchmark. Sample inputs can be found in the `/histogram_datafiles/` directory, which were gathered from the Phoenix [GitHub](https://github.com/fasiddique/DRAMAP-Phoenix/tree/main) ([direct download link](http://csl.stanford.edu/~christos/data/histogram.tar.gz)), with the execepton of `sample1.bmp`, which came from [FileSamplesHub](https://filesampleshub.com/format/image/bmp). Additional files that exceeded the file size for GitHub which were used in benchmarks for the paper can be found in the following Google Drive [folder](https://drive.google.com/drive/u/3/folders/1sKFcEftxzln6rtjftChb5Yog_9S5CDRd).

## Directory Structure

```
cpp-histogram/
├── PIM/
│   ├── Makefile
│   ├── hist.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── hist.cpp
│   ├── GPU/
│   │   ├── Makefile
│   │   ├── hist.cu
├── histogram_datafiles/
│   ├── sample1.bmp
│   ├── small.bmp
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the histogram benchmark:

1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The CPU variant of histogram has been implemented using standard C++ library and pthread for parallel execution. Other C++ libraries, such as OpenBLAS or Boost, were not used as either histogram implementations did not exist or did fall not fall into the scope of our use case. Threading was chosen in favor of OpenMP as timings had no significant differences, and due to previous works, such as Phoenix, which we modeled our benchmark after and modified slightly to make the method you see.

#### GPU

The GPU variant leverages CUDA C++ Core Libaries (CCCL), specifically CUB and its `DeviceHistogram` API, to perform a histogram algorithm of a 24-bit .bmp file on NVIDIA GPU.

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

*Note that the GPU Makefile currently uses SM_80, which is compatible with the A100. To run it on a different GPU, please manually change this in the makefile.

### PIM Variant

To compile for the PIM variant, use:

```bash
cd PIM
make
```

## Execution Instructions

### Running the Executable

After compiling, run each executable with the following command that will run it with default parameters:

```bash
./hist.out
```

To see help text on all usages, use following command:

```bash
./hist.out -h
```

### Specifying Input File

By default, `sample1.bmp` from the `/histogram_datafiles/` directory is used as the input file; however, you can specify a valid .bmp file using the `-i` flag:

```bash
./hist.out -i <input_file>
```

## References

<a id = "1">[1]</a>
Colby Ranger, Ramanan Raghuraman, Arun Penmetsa, Gary Bradski,
and Christos Kozyrakis. Evaluating mapreduce for multi-core and
multiprocessor systems. In 2007 IEEE 13th International Symposium
on High Performance Computer Architecture, pages 13–24. Ieee, 2007.
