# VGG19 

VGG19 is a deep convolutional neural network architecture with 13 convolutional layers and 3 fully connected layers designed for image classification and feature extraction with a focus on simplicity and depth. Each convolutional layer and fully connected layer (except the last one) is followed by an ReLU activation function to introduce non-linearity. There is also a max pool layer after the 2nd, 4th, 8th, 12th and 16th convolutional layer for a total of 5 max pooling layers.

## Directory Structure
```
cpp-vgg19/
├── PIM/
│   ├── Makefile
│   ├── vgg19.cpp
│   ├── vgg19_weights.py
├── baselines/
│   ├── vgg19.py
│   ├── categories.txt
│   ├── data
|       |- test/ 
├── README.md
├── Makefile
├── slurm.h
```

## Implementation Description

This repository contains two different implementations of the VGG19 benchmark:
1. CPU & GPU
2. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU & GPU

* The CPU and GPU variants of VGG19 have been implemented using standard Python and PyTorch framework for machine learning. The same script is used for both, the device is specified from the command line by the user. The script performs image classification using a pre-trained VGG19 model, supporting both CPU and CUDA-enabled GPU devices. It loads and preprocesses images from a specified directory (default: ../../cpp-vgg13/baselines/data/test/), then performs inference to classify the images, outputting the top 5 predicted categories with their probabilities. The test directory currently has 5 images which consume a total space of ~700 KB.
* Since the model is trained on ImageNet dataset, the output from the softmax layer has 1000 classes. The 1000 classes are specified in a categories.txt file (default: ../../cpp-vgg13/baselines/categories.txt). The script then uses the categories.txt file to determine the top 5 results. Results, including the top 5 results with the labels and their probabilities and execution time in ms, are printed at the end. Command-line arguments specify the device, and cpu is chosen as the default device. If the device is specified as cuda, and cuda is not available, cpu is chosen as the fallback option.

### PIM Implementation

The PIM variant is implemented using C++ and three different PIM architectures - BITSIMD, Fulcrum and Bank-level can be tested with this. 
  
## Compilation and Execution Instructions for Specific Variants

### CPU Variant

To run the script for the CPU variant, use command like the following example:

```bash
cd baselines
python3 vgg19.py -cuda f -c categories.txt -d data/test/
```
Note: 
 * "-cuda" to specify the device to use for inference. 't' -> cuda, 'f' -> cpu.
 * "-c" to specify the text file with the 1000 classes and their corresponding labels.
 * "-d" to specify the directory containing the images to be used for the inference.

### GPU Variant

To run the script for the GPU variant, use command like the following example:

```bash
cd baselines
python3 vgg19.py -cuda t -c categories.txt -d data/test/
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
./vgg19.out
```

### Specifying the Input Image and Kernels

The input JPEG image and kernel CSV file can be specified as in the following command:

```bash
./vgg19.out -i <input_image> -k <kernel_CSV_file> -v t  
```
Note: 
* To specify an input JPEG image, the C++ file must first be compiled by passing "COMPILE_WITH_JPEG=1" during make.
* To get the kernel CSV file from a pre-trained VGG19 model, run the vgg19_weights.py script like below:
  ```bash
  python3 vgg19_weights.py
  ```

### Help and Usage Options

For help or to see usage options, use the `-h` option:

```bash
./vgg19.out -h
```
