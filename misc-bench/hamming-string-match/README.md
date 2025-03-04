# Hamming String Match

Hamming string match takes one string (called the text), 1 or more additional strings (called the keys), and a nonnegative integer (called the maxHammingDistance), then finds all locations where the keys appear in the text. The keys are allowed to have a maximum of maxHammingDistance character substitutions from the text while still finding a match. This project uses an output format derived from the [PFAC](https://github.com/pfac-lib/PFAC/) project, which is described below:
- The output is an array of the same length as the text
- The elements of the output array are 32-bit ints
- Each element represents if there was a match with a key at that position in the text
    - 0 in the output represents no match at the given position
    - A non-zero value in the output means that the key with that index matches at that position (indices start at 1)
- If multiple keys match for the same position, the key with the larger index is given priority

For example:
- text = "ABCAEFG"
- keys = ["WW", "ABC"]
- maxHammingDistance = 2
- hammingStringMatch(text, keys, maxHammingDistance) -> [3, 1, 1, 3, 1, 1, 0]
- Explanation:
    - "WW" is key 1, "ABC" is key 2
    - "WW" matches everywhere, since its length is less than or equal to maxHammingDistance
        - Does not match at the last position because it would go past the end of the text
    - "ABC" matches at the first position with 0 substitutions and at position 3 with 2 substitutions
        - The matches for "ABC" takes priority because its index is higher

## Directory Structure

```
hamming-string-match/
├── PIM/
│   ├── Makefile
│   ├── hamming-string-match.cpp
│   ├── slurm.sh
├── hamming-data-generator/
│   ├── Makefile
│   ├── hamming-data-generator.cpp
├── dataset/
│   ├── 10mil_l-10_nk-10_kl/
│   │   ├── text.txt
│   │   ├── keys.txt
│   │   ├── maxHammingDistance.txt
├── README.md
├── Makefile
```

## Implementation Description

This repository contains one implementation of the Hamming string match benchmark:

1. PIM

### PIM Implementation

The PIM variant is implemented using C++ with some simulation speedup from OpenMP. Three different PIM architectures can be tested with this.

## Hamming String Matching Inputs

### Data Format

Running the string match algorithm requires a text file, a keys file, and a max Hamming distance file. The text file should consist of a string to be matched against. The keys file should contain the keys to be matched, with each key on a newline, as well as a blank line at the end of the file. Do not include duplicate keys. The Hamming distance file should contain one unsigned integer representing the maximum number of character substitutions a key can have while still matching part of the text.

### Data Generator Instructions

The data generator provides a way to create synthetic string matching data with specific desired properties. To compile the data generator, run:

```bash
cd hamming-data-generator
make
```

The data generator will write the generated text, keys, and Hamming distance to `dataset/<output folder name>/text.txt`, `dataset/<output folder name>/keys.txt`, and `dataset/<output folder name>/maxHammingDistance.txt`, respectively. To run the data generator, use the following:

```bash
cd hamming-data-generator
./hamming-data-generator.out -o <output folder name>
```

To see help text on all usages and how to modify any of the input parameters, use the following command:

```bash
./hamming-data-generator.out -h
```

## Compilation Instructions

To compile the PIM Hamming string match kernel, use:

```bash
cd PIM
make
```

## Execution Instructions

### Running the Executable

After compiling, run each executable with the following command. Note that the default keys file is `dataset/10mil_l-10_nk-10_kl/keys.txt`, the default text file is `dataset/10mil_l-10_nk-10_kl/text.txt`, and the default Hamming distance file is `dataset/10mil_l-10_nk-10_kl/maxHammingDistance.txt`, but these can be changed using `-k`, `-t`, and `-d`, respectively.

```bash
./hamming-string-match.out
```

To see help text on all usages and how to modify any of the input parameters, use the following command:

```bash
./hamming-string-match.out -h
```

## References
Cheng-Hung Lin, Chen-Hsiung Liu, Lung-Sheng Chien, Shih-Chieh Chang, "Accelerating Pattern Matching Using a Novel Parallel Algorithm on GPUs," IEEE Transactions on Computers, vol. 62, no. 10, pp. 1906-1916, Oct. 2013, doi:10.1109/TC.2012.254
