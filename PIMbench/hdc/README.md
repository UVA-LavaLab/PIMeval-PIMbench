# Hyperdimensional Computing (HDC) 
Hyperdimensional Computing (HDC) is a brain-inspired computational paradigm that represents data using high-dimensional vectors and performs efficient, lightweight operations for classification and pattern recognition tasks. HDC is designed to be robust against noise and scalable across various applications, making it suitable for low-power and parallel processing architectures.

Each input data point is encoded into a high-dimensional vector using a predefined mapping strategy. These vectors are then processed through element-wise operations such as bundling, binding, and permutation to learn patterns. Classification is performed by comparing query vectors to stored class prototypes using similarity measures. 


## Directory Structure 
```
hdc
|-- Dataset
|   |-- human_yeast_targetdecoy_vec_34976.charge2.npz
|   |-- iPRG2012_vec_34976.charge2.npz
|   `-- oms
|-- PIM
|   |-- Makefile
|   |-- hdc.cpp
|   |-- hdc.out
|   `-- slurm.sh
|-- apptainer.def
|-- apptainer.sif
`-- baselines
    |-- hd_oms.py
    |-- model.py
    |-- slurm.sh
    `-- utils.py
```

## Implementation Description 
This repository contains two different implementations of the HDC benchmark:
1. CPU & GPU based on Pytorch
2. PIM

### Baseline Implementations 

* The CPU and GPU variants of the **Hyperdimensional Computing (HDC) database search** have been implemented using standard Python and PyTorch. The same script is used for both, and the device is specified from the command line by the user. The script performs similarity search using HDC-encoded reference and query vectors, supporting both CPU and CUDA-enabled GPU devices.
* The script first loads reference and query datasets from specified files, encodes them into high-dimensional vectors using an HDC model, and then performs similarity-based database search. 
* By default, both CPU and GPU run with a batch size of **8192**, but users can specify different batch sizes for CPU and GPU from the command line. If the number of query samples is smaller than the batch size, existing queries are reused to ensure consistency in processing.
* The similarity search is performed using matrix multiplication between query and reference encodings, followed by a ranking step to identify the **top-k** most similar results. The `top-k` value (default: 5) can be adjusted via command-line arguments.

### PIM Implementation 
The PIM variant is implemented using C++ and three different PIM architectures - BITSIMD, Fulcrum and Bank-level can be tested with this.

## Compilation Instructions for Specific Variants
### PIM Variant 
To compile for the PIM variant, use:

```bash
cd PIM
make
```

## Execution Instructions 
### CPU Variant
To run the script for the **CPU variant**, use the following command:

```bash
cd baselines
python3 hd_oms.py --ref-dataset hcd --n-ref-test 8192 --n-query-test 8192 --topk 5
```

**Note:**
- `--ref-dataset` specifies the dataset to use. Options: `iprg` (iPRG2012 dataset) or `hcd` (Massive Human HCD dataset).
- `--n-ref-test` and `--n-query-test` set the number of reference and query samples for evaluation.
- `--topk` specifies the number of top-k results to retrieve.
- The script defaults to **CPU execution** unless `--cuda` is specified.

---

### GPU Variant
To run the script for the **GPU variant**, use the following command:

```bash
cd baselines
python3 hd_oms.py --cuda --ref-dataset hcd --n-ref-test 8192 --n-query-test 8192 --topk 5
```

**Note:**
- `--cuda` enables GPU execution. If CUDA is not available, the script automatically falls back to CPU.
- The same dataset and parameter options as the CPU variant apply.

---

### Apptainer Execution
If running inside an **Apptainer container**, use the following command:

```bash
apptainer exec ../apptainer.sif python3 hd_oms.py --cuda --ref-dataset hcd --n-ref-test 8192 --n-query-test 8192 --topk 5
```

**Note:**
- `apptainer exec ../apptainer.sif` runs the script inside the specified container.
- The same execution parameters as the CPU and GPU variants apply.
- Ensure that `apptainer.sif` is correctly built and located in the expected directory before execution.

---

### **Additional Options**
- `--output-ref-file <path>`: Specify a custom output file for reference encodings.
- `--output-query-file <path>`: Specify a custom output file for query encodings.
- `--warmup <int>`: Set the number of warmup iterations before benchmarking.
- `--n-lv <int>`: Define the number of quantization levels for level HVs.
- `--n-dim <int>`: Set the hyperdimensional vector dimension (default: 8192).

### PIM Variant

After compiling, run the HDC PIM variant executable with the following command, which will run it with default parameters:

```bash
./hdc.out
```

---

#### Specifying Input Files and Parameters
The reference and query files, as well as other parameters, can be specified using the following command:

```bash
./hdc.out -r <input_reference_file> -q <input_query_file> -k <top_k> -d <hypervector_dimension> -R <num_reference_samples> -Q <num_query_samples> -c <dramsim_config_file>
```

**Note:**
- `-r` specifies the **input reference file** (optional in random input mode).
- `-q` specifies the **input query file** (optional in random input mode).
- `-k` sets the **top-k** value for similarity search (**default: 5**).
- `-d` defines the **hypervector dimension**, which can range from **1K to 16K** (**default: 8192**).
- `-R` sets the **number of reference samples** (**default: 8192**).
- `-Q` sets the **number of query samples** (**default: 8192**).
- `-c` specifies the **input file containing DRAMSim configuration**.

---

#### Verifying Results with CPU
To verify the HDC PIM execution results against CPU-based computation, use:

```bash
./hdc.out -v
```

This option ensures the results computed on the **PIM hardware** are checked against the **CPU implementation**.

---

#### Help and Usage Options
For help or to see usage options, use the `-h` flag:

```bash
./hdc.out -h
```



