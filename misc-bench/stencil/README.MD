# Stencil

Iterative stencil loops consist of repeatedly applying a stencil pattern to a grid. This benchmark iterates a stencil average pattern on a 2D array, with both the number of iterations and the size of the stencil pattern parameterized. An example is below:

- Input: [[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], radius=1, iterations=1
- Output: [[#, #, #], [#, 1.66, #], [#, #, #]]
- \# represents an undefined element

## Directory Structure

```
stencil/
├── PIM/
│   ├── Makefile
│   ├── stencil.cpp
├── README.md
├── Makefile
```

## Implementation Description

This repository contains one implementation of the stencil benchmark:

1. PIM

### PIM Implementation

The PIM variant is implemented using C++ with some speedup from OpenMP. Three different PIM architectures can be tested with this.

## Compilation Instructions for Specific Variants

### PIM Variant

To compile for the PIM variant, use:

```bash
cd PIM
make
```

## Execution Instructions

### Running the Executable

After compiling, run each executable with the following command that will run it for default parameters:

```bash
./stencil.out
```

To see help text on all usages and how to modify any of the input parameters, use the following command:

```bash
./stencil.out -h
```
