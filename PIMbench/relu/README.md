# ReLU 

ReLU (Rectified Linear Unit) is an activation function used in deep learning that applies an element-wise operation to the input tensor, setting all negative values to zero while keeping positive values unchanged. This non-linear operation introduces sparsity in the network by allowing only positive activations to pass through, which helps in mitigating the vanishing gradient problem. The simplicity of ReLU makes it computationally efficient and a popular choice in convolutional neural networks (CNNs). ReLU enhances the model's ability to learn complex patterns by enabling it to approximate non-linear functions effectively.

## Directory Structure
```
cpp-relu/
├── PIM/
│   ├── Makefile
│   ├── relu.cpp
├── baselines/
│   ├── relu.py
├── README.md
├── Makefile
```

## Implementation Description

This repository contains two different implementations of the ReLU benchmark:
1. CPU & GPU
2. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU & GPU

* The CPU and GPU variants of ReLU have been implemented using standard Python and PyTorch framework for machine learning. The same script is used for both, the device is specified from the command line by the user. The script performs ReLU operation in batches, supporting both CPU and CUDA-enabled GPU devices. It loads input matrices in batches, then performs ReLU for each element in the input tensor.  
* GPU and CPU by default have a batch size of 64, and the batch size can be specified differently for CPU and GPU from the command line. 
* The total execution time for the complete batch, in ms, is printed at the end. Command-line arguments specify the device, and cpu is chosen as the default device. If the device is specified as cuda, and cuda is not available, cpu is chosen as the fallback option.

### PIM Implementation

The PIM variant is implemented using C++ and three different PIM architectures - BITSIMD, Fulcrum and Bank-level can be tested with this. 
  
## Compilation and Execution Instructions for Specific Variants

### CPU Variant

To run the script for the CPU variant, use command like the following example:

```bash
cd baselines
python3 relu.py -b 64 -d 64 -r 224 -c 224 
```
Note: 
 * "-b" to specify the batch size for the input.
 * "-d" to specify the depth of the input matrix.
 * '-r" to specify the number of rows in the input matrix.
 * "-c" to specify the number of columns in the input matrix.

### GPU Variant

To run the script for the GPU variant, use command like the following example:

```bash
cd baselines
python3 relu.py -cuda 
```
Note: 
 * "-cuda" is specified to use GPU for the batch ReLU. Default -> CPU.
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
./relu.out -d 128 -r 226 -c 226 -v t
```
Note: 
 * "-d" to specify the depth of the input matrix.
 * '-r" to specify the number of rows in the input matrix.
 * "-c" to specify the number of columns in the input matrix. 
 * "-v t" to compare the PIM results with CPU results and print the mismatches. Default -> not compared.

### Help and Usage Options

For help or to see usage options, use the `-h` option:

```bash
./relu.out -h
```
