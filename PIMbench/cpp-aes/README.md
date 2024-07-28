# Advanced Encryption Standard (AES) in ECB 256 Mode

This project implements the Advanced Encryption Standard (AES) in ECB 256 mode using CPU, GPU, and PIM variants.

## Directory Structure

```
./cpp-aes/
├── baselines
│   ├── CPU
│   │   ├── aes.cpp
│   │   └── Makefile
│   └── GPU
│       ├── aes.cu
│       └── Makefile
├── Makefile
├── PIM
│   ├── aes.cpp
│   ├── aes.out
│   ├── Makefile
│   ├── PIMAuxilary.cpp
│   └── PIMAuxilary.h
└── slurm.sh
```

## Implementation Description

This repository contains three different implementations of the AES encryption in ECB 256 mode:
1. CPU
2. GPU
3. PIM

### Baseline Implementation

The CPU and GPU implementations serve as baseline comparisons for the PIM implementation.

#### CPU

The CPU variant of AES encryption is implemented using standard C++.

#### GPU

The GPU variant utilizes CUDA to perform AES encryption on NVIDIA GPUs.

### PIM Implementation

The PIM variant is implemented using C++ and includes auxiliary files to support the PIM architecture.

## Compilation Instructions for Specific Variants

### CPU Variant

To compile the CPU variant, use:

```bash
cd baselines/CPU
make
```

### GPU Variant

To compile the GPU variant, use:

```bash
cd baselines/GPU
make
```
* **NOTE**: Make sure to install nvcc for GPU code compilation.

### PIM Variant

To compile the PIM variant, use:

```bash
cd PIM
make
```

## Usage

To run the AES encryption executable, use the following syntax:

```bash
./aes.out [options]
```

### Options

- `-l`: Input size (default=65536 bytes)
- `-k`: Key file containing two vectors (default=generates key with random numbers)
- `-i`: Input file containing two vectors (default=generates input with random numbers)
- `-c`: Cipher file containing two vectors (default=./cipher.txt)
- `-o`: Output file containing two vectors (default=./output.txt)
- `-v`: (true/false) Validates if the input file and output file are the same (default=false)

### Running the Executable

After compiling, run the executable with the desired options. For example, to run with default parameters:

```bash
./aes.out
```

To specify an input size and validate the output:

```bash
./aes.out -l 1024 -v true
```

## Example Usage

Here’s an example to run AES encryption with a specified input size and validate the output:

```bash
./aes.out -l 2048 -i ./input.txt -k ./key.txt -c ./cipher.txt -o ./output.txt -v true
```

This command will:
- Encrypt data from `input.txt` using the key from `key.txt`
- Store the cipher text in `cipher.txt`
- Store the decrypted output in `output.txt`
- Validate if the input and output files are the same

## SLURM Job Submission

If you are using a SLURM workload manager, you can submit a job using the provided `slurm.sh` script. Modify the script as needed to suit your environment.

```bash
sbatch slurm.sh
```


