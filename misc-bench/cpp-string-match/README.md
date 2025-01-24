# String Match

String match takes one string (called the text), and 1 or more additional strings (called the keys), and finds all locations where the keys appear in the text. There are multiple different possible output formats for string matching, this project uses the output format from the [PFAC](https://github.com/pfac-lib/PFAC/) project, as described below:
- The output is an array of the same length as the text
- The elements of the output array are 32 bit ints
- Each elements represents if there was a match with a key at that position in the text
    - 0 in the output represents no match at the given position
    - A non-zero value in the output means that the key with that index matches at that position (indices start at 1)
- For keys that are prefixes of other keys, the longest matching key is always given as the result

For example:
- text = "ABCAB"
- keys = ["C", "AB", "ABC"]
- string_match(text, keys) -> [3, 0, 1, 2, 0]

## Directory Structure

```
cpp-string-match/
├── include/
│   ├── string-match-utils.h
├── PIM/
│   ├── Makefile
│   ├── string-match.cpp
├── baselines/
│   ├── CPU/
│   │   ├── CMakeLists.txt
│   │   ├── string-match-container.def
│   │   ├── .gitignore
│   │   ├── hyperscan/ (submodule)
│   │   ├── src/
│   │   │   ├── CMakeLists.txt
│   │   │   ├── string-match.cpp
│   ├── GPU/
│   │   ├── Makefile
│   │   ├── string-match.cu
│   │   ├── PFAC/ (submodule)
├── data-generator/
│   ├── Makefile
│   ├── data-generator.cpp
├── dataset/
│   ├── .gitkeep
├── README.md
├── Makefile
```

## Implementation Description

This repository contains three different implementations of the string match benchmark:

1. CPU
2. GPU
3. PIM

### Baseline Implementation

CPU and GPU have been used as baselines.

#### CPU

The CPU variant of string matching uses [hyperscan](https://github.com/intel/hyperscan) to match the strings.

#### GPU

The GPU variant leverages the [PFAC](https://github.com/pfac-lib/PFAC/) library.

### PIM Implementation

The PIM variant is implemented using C++ with some simlation speedup from OpenMP. Three different PIM architectures can be tested with this.

## String Matching Inputs

### Data Format

Running the string match algorithm requires both a text file and a keys file. Both files are read from the `dataset/` directory. The text file should consist of a string to be matched against. The keys file should contain the keys to be matched, with each key on a newline, as well as a blank line at the end of the file. Also note that keys are expected to be sorted in order of increasing length, and that duplicate keys will cause errors.

### Data Generator Instructions

The data generator provides a way to create synthetic string matching data with specific desired properties. To compile the data generator, run:

```bash
cd data-generator
make
```

The data generator will write the generated text and keys to `dataset/<output folder name>/text.txt` and `dataset/<output folder name>/keys.txt`, respectively. To run the data generator, use the following:

```bash
cd data-generator
./data-generator.out -o <output folder name>
```

To see help text on all usages and how to modify any of the input parameters, use following command:

```bash
./string-match.out -h
```

## Compilation Instructions for Specific Variants

### CPU Variant

If the submodules have not already been initialized, run:

```bash
git submodule update --init --recursive
```

The CPU varient requires [apptainer](https://apptainer.org/) to run the container to compile in (but can run outside of the container). To compile for the CPU variant, use:

```bash
cd baselines/CPU
apptainer build string-match.sif string-match-container.def
apptainer run string-match.sif
mkdir -p build
cd build
cmake ..
make
exit
```

### GPU Variant

If the submodules have not already been initialized, run:

```bash
git submodule update --init --recursive
```

Due to limitations of the [PFAC](https://github.com/pfac-lib/PFAC/) library, the GPU baseline must be compiled with a cuda version < 12.0.0. To compile for the GPU variant, first compile the PFAC library using:

```bash
cd baselines/GPU/PFAC/PFAC
make
```

Then compile the GPU string match baseline using:
```bash
cd baselines/GPU
export LD_LIBRARY_PATH=$(pwd)/PFAC/PFAC/lib:$LD_LIBRARY_PATH # Replace LD_LIBRARY_PATH with DYLD_LIBRARY_PATH for Mac
make
```

*Note that the GPU Makefile currently uses SM_80, which is compatible with the A100. To run it on a different GPU, please manually change this in the makefile.

### PIM Variant

To compile for the PIM variant, use:

```bash
cd PIM
make
```

## Execution Instructions

### Running the Executable

After compiling, run the each executable with the following command (Must be run in the build directory for the CPU baseline). Note that for the GPU baseline, the LD_LIBRARY_PATH enviornment variable (DYLD_LIBRARY_PATH on Mac) must include the PFAC lib directory, as set in the compilation instructions for the GPU varient. Both the -k and -t parameters are required, and specify the keys and the text to match, reading from `dataset/<keys file>` and `dataset/<text file>`:

```bash
./string-match.out -k <keys file> -t <text file>
```

To see help text on all usages and how to modify any of the input parameters, use following command:

```bash
./string-match.out -h
```

## References
Cheng-Hung Lin, Chen-Hsiung Liu, Lung-Sheng Chien, Shih-Chieh Chang, "Accelerating Pattern Matching Using a Novel Parallel Algorithm on GPUs," IEEE Transactions on Computers, vol. 62, no. 10, pp. 1906-1916, Oct. 2013, doi:10.1109/TC.2012.254