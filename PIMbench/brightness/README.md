# Brightness

Brightness is a basic image processing algorithm that increments the color channel components for all image pixels by a specified degree. The PIM, CPU, and GPU implementations all use the same process of accomplishing this. The adjustment is made by adding the brightness coefficient to each pixel of image data and peforming a check to see if the change goes out of bounds for valid pixel values. If out of bounds situations occur, the pixel is rounded to the nearest max/min value, such as 276 rounding down to 255 or -20 rounding up to 0. The inclusion of this benchmark was inspired by previous works, such as the SIMDRAM framework paper [[1]](#1).

For a detailed description of the Brightness algorithm, refer to the OpenCV brightness [documentation](https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_linear_transform.html).

## Input Files

As mentioned above, only 24-bit .bmp files can be used to test the functionality of this benchmark. Sample inputs can be found in the `/cpp-histogram/histogram_datafiles/` directory, which were gathered from the Phoenix [GitHub](https://github.com/fasiddique/DRAMAP-Phoenix/tree/main) ([direct download link](http://csl.stanford.edu/~christos/data/histogram.tar.gz)), with the execepton of `sample1.bmp`, which came from [FileSamplesHub](https://filesampleshub.com/format/image/bmp). Additional files that exceeded the file size for GitHub which were used in benchmarks for the paper can be found in the following Google Drive [folder](https://drive.google.com/drive/u/3/folders/1sKFcEftxzln6rtjftChb5Yog_9S5CDRd).

## Directory Structure

```
cpp-brightness/
├── PIM/
│   ├── Makefile
│   ├── brightness.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── brightness.cpp
│   ├── GPU/
│   │   ├── Makefile
│   │   ├── brightness.cu
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

The CPU variant of histogram has been implemented using standard C++ library and OpenMP for parallel execution. This method was chosen as it was seen to be the most effective in parallelizing the independent execution operations. 

#### GPU

The GPU variant leverages CUDA C++ Core Libaries (CCCL), specifically Thrust and the `transform` API, to perform a histogram algorithm of a 24-bit .bmp file on NVIDIA GPU.

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
./brightness.out
```

To see help text on all usages, use following command:

```bash
./brightness.out -h
```

### Specifying Input File

By default, `sample1.bmp` from the `cpp-histogram/histogram_datafiles/` directory is used as the input file; however, you can specify a valid .bmp file using the `-i` flag:

```bash
./brightness.out -i <input_file>
```

### Specifying Brightness Coefficient

If you want to change the brightness coefficient (i.e. the amount the color channel values change by), use the `-b` option:

```bash
./brightness.out -b <value>
```

## References

<a id = "1">1.</a>
Nastaran Hajinazar, Geraldo F. Oliveira, Sven Gregorio, João Dinis Ferreira, Nika Mansouri Ghiasi, Minesh Patel, Mohammed Alser, Saugata Ghose, Juan Gómez-Luna, and Onur Mutlu. 2021. SIMDRAM: a framework for bit-serial SIMD processing using DRAM. In Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS '21). Association for Computing Machinery, New York, NY, USA, 329–345. https://doi.org/10.1145/3445814.3446749