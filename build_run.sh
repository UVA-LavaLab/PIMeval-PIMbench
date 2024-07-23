#!/bin/bash

# Function to build and run for a specific target
build_and_run() {
    local target=$1
    local output_file=$2

    # Build the simulator and benchmarks, redirect output to the specified file
    echo "Building target $target" > "$output_file" 2>&1
    make clean >> "$output_file" 2>&1
    make -j PIM_SIM_TARGET="$target" >> "$output_file" 2>&1

    # Function to recursively find and run .out files
    run_out_files() {
        for entry in "$1"/*; do
            if [ -d "$entry" ]; then
                # If the entry is a directory, recurse into it
                run_out_files "$entry"
            elif [ -f "$entry" ] && [[ "$entry" == *.out ]]; then
                # If the entry is a .out file, execute it
                echo "Running $(basename "$entry") for target $target" >> "$output_file" 2>&1
                pushd "$(dirname "$entry")" > /dev/null
                output=$(./"$(basename "$entry")" 2>&1)
                result=$?
                popd > /dev/null
                echo "$output" >> "$output_file" 2>&1
                if [ $result -ne 0 ]; then
                    echo "Benchmark $(basename "$entry") for target $target failed with error:" >> "$output_file" 2>&1
                    echo "$output" >> "$output_file" 2>&1
                fi
            fi
        done
    }

    # Check if PIMbench directory exists
    if [ -d "PIMbench" ]; then
        # Start the script from the PIMbench directory
        run_out_files "$(pwd)/PIMbench"
    else
        echo "Directory 'PIMbench' does not exist." >> "$output_file" 2>&1
    fi
}

# Define targets and their corresponding output files
declare -A targets
targets=(
    ["PIM_DEVICE_BITSIMD_V_AP"]="bitserial_ap.txt"
    ["PIM_DEVICE_FULCRUM"]="fulcrum.txt"
    ["PIM_DEVICE_BANK_LEVEL"]="bank_level.txt"
)

# Iterate over the targets and run the build and execution process for each
for target in "${!targets[@]}"; do
    build_and_run "$target" "${targets[$target]}"
done

