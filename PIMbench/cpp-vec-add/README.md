# Vector Addition (VA)

Vector addition implementation using C++. Supports parallel execution using OpenMP.

## Compilation Instructions

To compile with the default data type (`int32_t`), simply run:

```bash
make 
```
To specify a different data type (e.g., float or double), use the DATA_TYPE variable during compilation:

```bash
make DATA_TYPE=float
```

## Execution Instructions

After compiling, run the executable using the following command:

```bash
./vec_add.out
```

You can also specify the input size using the -i option:

```bash
./vec_add.out -i <input_size>
```
If you need help or want to see usage options, use the -h option:

```bash
./vec_add.out -h
```