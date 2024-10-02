# Image Downsampling

Image Downsampling is the process of making an image smaller, and is useful in various machine learning workflows.
This project benchmarks a specific type, called mipmapping or box-filtering, which sets each pixel in the output image to the average of a rectangle of pixels in the input image.
The following equation shows an expression of box filtering, for the case of scaling down the image to half the width and height:

y[i][j] 
$\leftarrow $
(x[2*i][2*j]>>2) + (x[2*i+1][2*j]>>2) + (x[2*i][2*j+1]>>2) + (x[2*i+1][2*j+1]>>2)

where:
- $x$ is a color channel of the input image as a matrix
- $y$ is the color channel of the output image as a matrix

## Directory Structure
```
cpp-image-downsampling/
├── Dataset/
│   ├── input_1.bmp
│   ├── input_2.bmp
│   ├── input_3.bmp
│   ├── input_4.bmp
├── PIM/
│   ├── Makefile
│   ├── image-downsampling.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── image-downsampling.cpp
│   ├── GPU/
│   │   ├── Makefile
│   │   ├── image-downsampling.cu
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the image downsampling benchmark:
1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The CPU variant of vector addition has been implemented using standard C++.

#### GPU

The GPU variant has been implemented using CUDA to perform box filtering of an image on NVIDIA GPU.

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
./image-downsampling.out
```

To see help text on all usages, use following command
```bash
./image-downsampling.out -h
```

### Specifying Input Image

You can specify an input image from the Dataset directory (BMP only) using the `-i` option:

```bash
./image-downsampling.out -i <input_file.bmp>
```

### Specifying Output Image

You can specify an output filename to write the downsampled image to using the `-o` option:

```bash
./image-downsampling.out -o <output_file.bmp>
```
