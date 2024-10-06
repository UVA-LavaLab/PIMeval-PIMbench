#!/bin/bash
# Run PIMeval Functional Testing
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

# README for PIMeval developers
# Before committing code to libpimeval/, please run this function testing
# script to catch any PIMeval simulator behavior changes.

# STEP 1: Collect PIMeval functional simulation outputs of different
#         PIM architectures into result_local.txt

# Get the location of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL="$SCRIPT_DIR/result-local.txt"
GOLDEN="$SCRIPT_DIR/result-golden.txt"

echo "##################################" | tee $LOCAL
echo "PIMeval Functional Testing Results" | tee -a $LOCAL
echo "##################################" | tee -a $LOCAL

export PIMEVAL_TARGET=PIM_DEVICE_BITSIMD_V_AP
$SCRIPT_DIR/test-functional.out | tee -a $LOCAL

export PIMEVAL_TARGET=PIM_DEVICE_FULCRUM
$SCRIPT_DIR/test-functional.out | tee -a $LOCAL

export PIMEVAL_TARGET=PIM_DEVICE_BANK_LEVEL
$SCRIPT_DIR/test-functional.out | tee -a $LOCAL


# STEP 2: Compare result_local.txt with result_golden.txt
#         Catch any differences between the two

if diff "$GOLDEN" "$LOCAL" > /dev/null; then
    echo
    echo "########################################################################################"
    echo "PIMeval Functional Testing >>>>> PASSED"
    echo "All results are identical. Existing PIMeval behavior is well preserved."
    echo "Congratulations!"
    echo "########################################################################################"
    echo
else
    echo
    echo "########################################################################################"
    echo
    diff "$GOLDEN" "$LOCAL"
    echo
    echo "########################################################################################"
    echo "PIMeval Functional Testing >>>>> FAILED !!!!!"
    echo "Warning: Your local code changes are affecting PIMeval outputs."
    echo "- result-golden.txt : reference outputs being tracked with git"
    echo "- result-local.txt : local outputs, do not commit"
    echo "This does not mean it's bad. Please review all diffs between the two files carefully."
    echo "- If diffs are expected, please update result-golden.txt and commit it with your changes"
    echo "- If diffs are not expected, please debug it further"
    echo "########################################################################################"
    echo
fi

