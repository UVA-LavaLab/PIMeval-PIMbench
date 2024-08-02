# VGG16 

VGG16 is a deep convolutional neural network architecture with 13 convolutional layers and 3 fully connected layers designed for image classification and feature extraction with a focus on simplicity and depth. Each convolutional layer and fully connected layer (except the last one) is followed by an ReLU activation function to introduce non-linearity. There is also a max pool layer after the 2nd, 4th, 7th, 10th and 13th convolutional layer for a total of 5 max pooling layers.

## Directory Structure
```
cpp-vgg16/
├── PIM/
│   ├── Makefile
│   ├── vgg16.cpp
│   ├── vgg16_weights.py
├── baselines/
│   ├── vgg16.py
│   ├── categories.txt
│   ├── data
|       |- test/ 
├── README.md
├── Makefile
├── slurm.h
```

## Implementation Description

This repository contains two different implementations of the VGG16 benchmark:
1. CPU & GPU
2. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU & GPU

* The CPU and GPU variants of VGG16 have been implemented using standard Python and PyTorch framework for machine learning. The same script is used for both, the device is specified from the command line by the user. The script performs image classification in batches using a pre-trained VGG16 model, supporting both CPU and CUDA-enabled GPU devices. It loads and preprocesses images from a specified directory (default: ../../cpp-vgg13/baselines/data/test/) in batches, then performs inference to classify the images, outputting the top 5 predicted categories with their probabilities. The test directory currently has 5 images which consume a total space of ~700 KB. 
* GPU and CPU by default have a batch size of 64, and the batch size can be specified differently for CPU and GPU from the command line. If the number of images in the test directory is lower than the batch size, the existing images are replicated so that the number of images equals the batch size.
* Since the model is trained on ImageNet dataset, the output from the softmax layer has 1000 classes. The 1000 classes are specified in a categories.txt file (default: ../../cpp-vgg13/baselines/categories.txt). The script then uses the categories.txt file to determine the top 5 results. Results, including the top 5 results with the labels and their probabilities and execution time per image in ms, are printed at the end. Command-line arguments specify the device, and cpu is chosen as the default device. If the device is specified as cuda, and cuda is not available, cpu is chosen as the fallback option.

### PIM Implementation

The PIM variant is implemented using C++ and three different PIM architectures - BITSIMD, Fulcrum and Bank-level can be tested with this. 
  
## Compilation and Execution Instructions for Specific Variants

### CPU Variant

To run the script for the CPU variant, use command like the following example:

```bash
cd baselines
python3 vgg16.py -c categories.txt -d data/test/ 
```
Note: 
 * "-c" to specify the text file with the 1000 classes and their corresponding labels.
 * "-d" to specify the directory containing the images to be used for the inference.
 
### GPU Variant

To run the script for the GPU variant, use command like the following example:

```bash
cd baselines
python3 vgg16.py -cuda -c categories.txt -d data/test/
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

After compiling, run the executable with the following command that will run it for default parameters:

```bash
./vgg16.out
```

### Specifying the Input Image and Kernels

The input JPEG image and kernel CSV file can be specified as in the following command:

```bash
./vgg16.out -i <input_image> -k <kernel_CSV_file> -v t  
```
Note: 
* To specify an input JPEG image, the C++ file must first be compiled by passing "COMPILE_WITH_JPEG=1" during make.
* To get the kernel CSV file from a pre-trained VGG16 model, run the vgg16_weights.py script like below:
  ```bash
  python3 vgg16_weights.py
  ```

### Help and Usage Options

For help or to see usage options, use the `-h` option:

```bash
./vgg16.out -h
```
