#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re

if len(sys.argv) != 2:
    print("Usage:")
    print("./bitSerial.out > result.txt")
    print("./parseResult.py result.txt")
    exit()

filename = sys.argv[1]
print("INFO: Bit-serial Micro-program Performance Results")

with open(filename, 'r') as f:
    device = ''
    data_type = ''
    op = ''
    numR = 0
    numW = 0
    numL = 0
    for line in f:
        line = line.strip()

        match1 = re.match(r'\[(.*):(.*):(.*):(.*)\] Start', line)
        if match1:
            prev_device = device
            device = match1.group(1)
            if device != prev_device:
                if prev_device:
                    print("  }},")
                print("  { PIM_DEVICE_%s, {" % (device.upper()))
            prev_data_type = data_type
            data_type = match1.group(2)
            if data_type != prev_data_type:
                if prev_data_type:
                    print("    }},")
                print("    { PIM_%s, {" % (data_type.upper()))
            op = match1.group(3)
            numR = 0
            numW = 0
            numL = 0

        match2 = re.match(r'Num Read, Write, Logic : (\S+), (\S+), (\S+)', line)
        if match2:
            numR = int(match2.group(1))
            numW = int(match2.group(2))
            numL = int(match2.group(3))

        match3 = re.match(r'\[(.*):(.*):(.*):(.*)\] End', line)
        if match3:
            print("      { PimCmdEnum::%-13s { %4d, %4d, %4d } }," % (op.upper() + ',', numR, numW, numL))

    print("    }")
    print("  }")


