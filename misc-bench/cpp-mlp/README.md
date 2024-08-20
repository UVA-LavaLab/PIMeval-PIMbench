# Multi-Layer Perceptron (MLP)

A multi-layer perceptron is a feedforward artifical neural network where each layer is fully connected. Additionally, each neuron has a nonlinear activation function.

## Directory Structure

```
cpp-mlp/
├── PIM/
│   ├── Makefile
│   ├── mlp.cpp
├── baselines/
│   ├── mlp.py
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the MLP benchmark:

1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU & GPU

* The CPU and GPU variants of MLP have been implemented using standard Python and PyTorch framework for machine learning. The same script is used for both, the device is specified from the command line by the user. The script performs batched inference using random inputs and weights.

### PIM Implementation

The PIM variant is implemented using C++. Three different PIM architectures can be tested with this.

## Compilation Instructions for Specific Variants

### CPU Variant

To run the script for the CPU variant, use the following example command:

```bash
cd baselines
python3 mlp.py -n N -l 30,60,45,5
```
Note: 
 * "-n" to specify the number of inference points in the batch.
 * "-l" to specify the configuration of the layers, where each number represents the number of neurons in each layer.
 
### GPU Variant

To run the script for the GPU variant, use the following example command:

```bash
cd baselines
python3 mlp.py -cuda
```
Note: 
 * "-cuda" is specified to use GPU for inference. Default -> CPU.
 * For GPU, it is assumed that the system has a GPU with CUDA support.

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
./mlp.out
```

### Help and Usage Options

For help or to see usage options, use the `-h` option:

```bash
./mlp.out -h
```
