# Convolution 

A convolution operation in deep learning is a mathematical process where a filter (kernel) slides over an input tensor, performing element-wise multiplications and summations at each position to produce an output tensor (feature map). This operation captures spatial hierarchies in the data by emphasizing specific features such as edges, textures, and patterns. The key parameters of a convolution include the kernel size, stride (step size of the filter movement), and padding (adding extra pixels around the input). convolutions are fundamental in convolutional neural networks (CNNs) for tasks like image recognition, where they help in automatically learning relevant features from input data.

## Directory Structure
```
cpp-conv/
├── PIM/
│   ├── Makefile
│   ├── conv.cpp
├── baselines/
│   ├── conv.py
├── README.md
├── Makefile
```

## Implementation Description

This repository contains two different implementations of the convolution benchmark:
1. CPU & GPU
2. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU & GPU

* The CPU and GPU variants of convolution have been implemented using standard Python and PyTorch framework for machine learning. The same script is used for both, the device is specified from the command line by the user. The script performs convolution operation in batches, supporting both CPU and CUDA-enabled GPU devices. It loads input matrcies in batches, then performs convolution with the kernel matrix in parallel based on the given stride and input padding.  
* GPU and CPU by default have a batch size of 64, and the batch size can be specified differently for CPU and GPU from the command line. 
* The total execution time for the complete batch, in ms, is printed at the end. Command-line arguments specify the device, and cpu is chosen as the default device. If the device is specified as cuda, and cuda is not available, cpu is chosen as the fallback option.

### PIM Implementation

The PIM variant is implemented using C++ and three different PIM architectures - BITSIMD, Fulcrum and Bank-level can be tested with this. 
  
## Compilation and Execution Instructions for Specific Variants

### CPU Variant

To run the script for the CPU variant, use command like the following example:

```bash
cd baselines
python3 conv.py -b 64 -d 64 -r 224 -c 224 -kr 3 -kc 3 -kd 64 -s 1 -p 1 
```
Note: 
 * "-b" to specify the batch size for the input.
 * "-d" to specify the depth of the input matrix.
 * '-r" to specify the number of rows in the input matrix.
 * "-c" to specify the number of columns in the input matrix.
 * "-kr" to specify the number of rows in the kernel matrix.
 * "-kc" to specify the number of columns in the kernel matrix.
 * "-kd" to specify the depth of the kernel matrix.
 * "-s" to specify the stride.
 * "-p" to specify the input padding. 
 
### GPU Variant

To run the script for the GPU variant, use command like the following example:

```bash
cd baselines
python3 conv.py -cuda 
```
Note: 
 * "-cuda" is specified to use GPU for the batch convolution. Default -> CPU.
 * For GPU, it is assumed that the system has a GPU with CUDA support.

### PIM Variant

To compile for the PIM variant, use:

```bash
cd PIM
make
```

## Execution Instructions

### Running the Executable

After compiling, run the executable like the following example command:

```bash
./conv.out -r 224 -c 224 -d 3 -k 3 -z 64 -p 1 -s 1 -v t
```
Note: 
 * "-d" to specify the depth of the input matrix.
 * '-r" to specify the number of rows in the input matrix.
 * "-c" to specify the number of columns in the input matrix.
 * "-k" to specify the number of rows and columns in the kernel matrix.
 * "-z" to specify the depth of the kernel matrix.
 * "-s" to specify the stride.
 * "-p" to specify the input padding. 
 * "-v t" to compare the PIM results with CPU results and print the mismatches. Default -> not compared.

### Help and Usage Options

For help or to see usage options, use the `-h` option:

```bash
./conv.out -h
```
