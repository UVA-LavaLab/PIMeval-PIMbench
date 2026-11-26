# Layer Normalization (LNORM)

The LayerNorm is a normalization function mostly used in AI models


For a detailed description of RMSNorm, you can refer to the [torch.nn.LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) or the [paper](https://arxiv.org/pdf/1607.06450)

## Directory Structure

```
rmsnorm/
├── PIM/
│   ├── Makefile
│   ├── lnorm.cpp
├── baselines/
│   ├── CPU/
│   │   ├── Makefile
│   │   ├── lnorm.cpp
│   ├── GPU/ **TODO**
│   │   ├── Makefile
│   │   ├── lnorm.cu 
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the RMSNORM benchmark:

1. CPU
2. GPU **TODO**
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The CPU variant ...

#### GPU

The GPU variant (**TODO** Try torch rmsnorm)

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
make -j USE_OPENMP=1
```

## Execution Instructions

### Running the Executable

After compiling, run the each executable with the following command that will run it for default parameters:

```bash
./lnorm.out
```

To see help text on all usages and how to modify any of the input parameters, use following command:

```bash
./lnorm.out -h
```
