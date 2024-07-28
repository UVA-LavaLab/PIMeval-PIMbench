# VGG13 

VGG13 is a deep convolutional neural network architecture with 10 convolutional layers and 3 fully connected layers designed for image classification and feature extraction with a focus on simplicity and depth. Each convolutional layer and fully connected layer (except the last one) is followed by an ReLU activation function to introduce non-linearity. There is also a max pool layer after every two convolutional layers for a total of 5 max pooling layers.

## Directory Structure
```
cpp-vgg13/
├── PIM/
│   ├── Makefile
│   ├── vgg13.cpp
│   ├── vgg13_weights.py
├── baselines/
│   ├── vgg13.py
├── README.md
├── Makefile
├── slurm.h
```

## Implementation Description

This repository contains two different implementations of the VGG13 benchmark:
1. CPU & GPU
2. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU & GPU

* The CPU and GPU variants of VGG13 have been implemented using standard Python and PyTorch framework for machine learning. The same script is used for both, the device is specified from the command line by the user. The script performs inference using a pre-trained VGG13 model on the CIFAR-100 dataset, supporting both CPU and CUDA-enabled GPU devices. The CIFAR-100 dataset is chosen because it is open-source and relatively small compared to other datasets. The entire dataset (both training and test sets) requires about 170 MB of storage. When downloaded, it should easily fit within the storage capacity of most modern systems. 
* The script preprocesses the data, adjusts the model's classifier for 100 classes, and optimizes it with TorchScript. The script evaluates the model's top-1 and top-5 accuracy and measures the execution time per image. Results, including accuracy percentages and execution time in ms, are printed at the end. Command-line arguments specify the device and the number of test images which is by default 1000.

### PIM Implementation

The PIM variant is implemented using C++ and three different PIM architectures - BITSIMD, Fulcrum and Bank-level can be tested with this. 
  
## Compilation and Execution Instructions for Specific Variants

### CPU Variant

To run the script for the CPU variant, use command like the following example:

```bash
cd baselines
python3 vgg13.py -s cpu -n 100
```
Note: 
 * "-s" to specify the device to use for inference (cpu/cuda).
 * "-n" to specify the number of images to be tested in the inferencing.

### GPU Variant

To run the script for the GPU variant, use command like the following example:

```bash
cd baselines
python3 vgg13.py -s cuda -n 100
```
Note: For GPU, it is assumed that the system has a GPU with CUDA support.

### PIM Variant

To compile for the PIM variant, use:

```bash
cd PIM
make
```

## Execution Instructions

### Running the Executable

After compiling, run the executable with the following command that will run it for default parameters:

```bash
./vgg13.out
```

### Specifying the Input Image and Kernels

The input JPEG image and kernel CSV file can be specified as in the following command:

```bash
./vgg13.out -i <input_image> -k <kernel_CSV_file> -v t  
```
Note: 
* To specify an input JPEG image, the C++ file must first be compiled by passing "COMPILE_WITH_JPEG=1" during make.
* To get the kernel CSV file from a pre-trained VGG13 model, run the vgg13_weights.py script like below:
  ```bash
  python3 vgg13_weights.py
  ```

### Help and Usage Options

For help or to see usage options, use the `-h` option:

```bash
./vgg13.out -h
```
