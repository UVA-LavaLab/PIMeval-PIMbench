#!/bin/bash

# Perform make clean and make in the current directory
echo "Running 'make clean'..."
make clean

echo "Running 'make'..."
make -j PIM_SIM_TARGET=PIM_DEVICE_FULCRUM

# Check if make was successful
if [ $? -ne 0 ]; then
    echo "'make' failed. Exiting."
    exit 1
fi

echo "Entering PIMbench directory"
cd PIMbench

# Iterate through each subdirectory in PIMbench and execute the benchmark
for dir in */; do
    if [ -d "$dir/PIM" ]; then
        # Change into the PIM directory
        cd "$dir/PIM"

        # Run the benchmark executable
        echo "Running benchmark in $dir"
        if [ -f *.out ]; then
            ./*.out
            # Check if the benchmark ran successfully
            if [ $? -ne 0 ]; then
                echo "Benchmark execution failed in $dir/PIM. Exiting."
                exit 1
            fi
        else
            echo "No executable found in $dir/PIM. Skipping."
        fi

        # Go back to the PIMbench directory
        cd ../../
    else
        echo "Running benchmark in $dir"
        cd "$dir/"
        ./*.out
        cd ../
    fi
done

echo "All benchmarks with small input completed!"
echo "Running vector add with large input:"

cd cpp-vec-add/PIM
./vec-add.out -l 1035544320
echo "Vector add with large input completed!"