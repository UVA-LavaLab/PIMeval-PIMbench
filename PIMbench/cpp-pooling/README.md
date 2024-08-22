# Max Pooling 

Max pooling is a down-sampling operation in deep learning that reduces the spatial dimensions of an input tensor by selecting the maximum value from a specified window (or filter) that slides over the input. This operation retains the most prominent features while discarding less important information, thereby reducing the computational load and helping to prevent overfitting. The key parameters of max pooling include the pool size (the dimensions of the window), stride (the step size of the window movement), and padding (adding extra pixels around the input). Max pooling is commonly used in convolutional neural networks (CNNs) to decrease the spatial dimensions of feature maps, making the network more efficient and focusing on the most critical features.

## Directory Structure
```
cpp-pooling/
├── PIM/
│   ├── Makefile
│   ├── pool.cpp
├── baselines/
│   ├── pool.py
├── README.md
├── Makefile
```

## Implementation Description

This repository contains two different implementations of the max pooling benchmark:
1. CPU & GPU
2. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU & GPU

* The CPU and GPU variants of max pooling have been implemented using standard Python and PyTorch framework for machine learning. The same script is used for both, the device is specified from the command line by the user. The script performs max pooling operation in batches, supporting both CPU and CUDA-enabled GPU devices. It loads input matrcies in batches, then performs max pooling based on the given window dimensions, stride and input padding.  
* GPU and CPU by default have a batch size of 64, and the batch size can be specified differently for CPU and GPU from the command line. 
* The total execution time for the complete batch, in ms, is printed at the end. Command-line arguments specify the device, and cpu is chosen as the default device. If the device is specified as cuda, and cuda is not available, cpu is chosen as the fallback option.

### PIM Implementation

The PIM variant is implemented using C++ and three different PIM architectures - BITSIMD, Fulcrum and Bank-level can be tested with this. 
  
## Compilation and Execution Instructions for Specific Variants

### CPU Variant

To run the script for the CPU variant, use command like the following example:

```bash
cd baselines
python3 pool.py -b 64 -d 64 -r 224 -c 224 -kh 2 -kw 2 -s 2 -p 0
```
Note: 
 * "-b" to specify the batch size for the input.
 * "-d" to specify the depth of the input matrix.
 * '-r" to specify the number of rows in the input matrix.
 * "-c" to specify the number of columns in the input matrix.
 * "-kh" to specify the height of the sliding window.
 * "-kw" to specify the width of the sliding window.
  * "-s" to specify the stride.
 * "-p" to specify the input padding. 
 
### GPU Variant

To run the script for the GPU variant, use command like the following example:

```bash
cd baselines
python3 pool.py -cuda 
```
Note: 
 * "-cuda" is specified to use GPU for the batch max pooling. Default -> CPU.
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
./pool.out -d 64 -r 224 -c 224 -l 2 -w 2 -s 2 -p 0 -v t
```
Note: 
 * "-d" to specify the depth of the input matrix.
 * '-r" to specify the number of rows in the input matrix.
 * "-c" to specify the number of columns in the input matrix.
 * "-l" to specify the height of the sliding window.
 * "-w" to specify the width of the sliding window.
 * "-s" to specify the stride.
 * "-p" to specify the input padding. 
 * "-v t" to compare the PIM results with CPU results and print the mismatches. Default -> not compared.

### Help and Usage Options

For help or to see usage options, use the `-h` option:

```bash
./pool.out -h
```
