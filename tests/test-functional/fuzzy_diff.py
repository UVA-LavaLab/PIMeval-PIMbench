#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python utility to compare PIMeval functional testing results
with Python2 and Python3 compatibility
Copyright (c) 2024 University of Virginia
This file is licensed under the MIT License.
See the LICENSE file in the root of this repository for more details.
"""

from __future__ import print_function
import difflib
import os
import sys

def is_equal(val1, val2, tolerance):
    """ Check if two tokens are float numbers within a tolerance """
    if val1 == val2:
        return True
    try:
        num1 = float(val1)
        num2 = float(val2)
        return abs(num1 - num2) <= tolerance
    except ValueError:
        return False

def process_chunk(chunk, tolerance):
    """ Process one diff chunk with @@ symbol """
    add = []
    sub = []
    for line in chunk:
        if line.startswith('+'):
            add.append(line)
        elif line.startswith('-'):
            sub.append(line)
    if len(add) != len(sub):
        return False
    for i in range(len(add)):
        tokens1 = add[i][1:].strip().split()
        tokens2 = sub[i][1:].strip().split()
        if len(tokens1) != len(tokens2):
            return False
        for j in range(len(tokens1)):
            if not is_equal(tokens1[j], tokens2[j], tolerance):
                return False;
    return True

def filter_diff_with_tolerance(diff, tolerance):
    """ Filter out diffs due to small FP errors """
    filtered_diff = []
    diff = list(diff)
    chunk = []
    for i in range(len(diff)):
        if i < 2:  # skip the first two --- and +++ lines
            continue
        chunk.append(diff[i].rstrip())
        if i + 1 == len(diff) or diff[i + 1].startswith('@@'):
            matched = process_chunk(chunk, tolerance)
            if not matched:
                filtered_diff += chunk
            chunk = []

    return filtered_diff

def compare_files_with_tolerance(file1, file2, tolerance):
    """ Compare two files with a small FP error tolerance """
    if not os.path.isfile(file1):
        print('Error: Invalid input file', file1)
        return 2
    if not os.path.isfile(file2):
        print('Error: Invalid input file', file2)
        return 2
    lines1 = []
    with open(file1, 'r') as f1:
        lines1 = f1.readlines()
    lines2 = []
    with open(file2, 'r') as f2:
        lines2 = f2.readlines()

    # perform diff
    diff = difflib.unified_diff(lines1, lines2, fromfile=file1, tofile=file2, lineterm='', n=0)

    # filter out diffs due to small FP error within tolerance
    filtered_diff = filter_diff_with_tolerance(diff, tolerance)

    # show outputs
    for line in filtered_diff:
        print(line)

    return 0 if len(filtered_diff) == 0 else 1

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: fuzzy_diff.py <file1> <file2>')
        print('This script diffs two files with a small FP error tolerance')
        print('Diffs are shown as outputs, otherwise empty')
        exit()

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    tolerance = 1.5e-6
    retVal = compare_files_with_tolerance(file1, file2, tolerance)
    sys.exit(retVal)  # 0 same, 1 diff, 2 error

