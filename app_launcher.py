#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PIMeval App Launcher

This is a helper script to launch PIMeval functional simulation applications with different configurations.
Please run with --help to see the available options.

Example: Running vec-mul of data size 10000 on BitSIMD-V-AP with 8 ranks:
./app_launcher.py 1 ./misc-bench/cpp-vec-mul/vec-mul.out -l 10000 --ra 8

Copyright (c) 2024 University of Virginia
This file is licensed under the MIT License.
See the LICENSE file in the root of this repository for more details.
"""

from __future__ import print_function
import sys
import argparse
import os

def main():
    device_names = [
        "PIM_DEVICE_BITSIMD_V",
        "PIM_DEVICE_BITSIMD_V_AP",
        "PIM_DEVICE_FULCRUM",
        "PIM_DEVICE_BANK_LEVEL"
    ]
    device_help = "Device ID mappings (sim_target):\n"
    for i, name in enumerate(device_names):
        device_help += "  %d: %s\n" % (i, name)
    parser = argparse.ArgumentParser(description="PIMeval App Launcher", epilog=device_help, formatter_class=argparse.RawTextHelpFormatter, add_help=False)
    # Re-add --help but not -h which is forwarded to executables
    parser.add_argument( "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit.")
    parser.add_argument("sim_target", type=int, help="PIMeval functional simulation target device ID")
    parser.add_argument("executable", type=str, help="Path to the executable to run")
    parser.add_argument("--ra", type=int, help="Number of ranks", default=1, metavar="<1>")
    parser.add_argument("--ba", type=int, help="Number of banks per rank", default=128, metavar="<128>")
    parser.add_argument("--sa", type=int, help="Number of subarrays per bank", default=32, metavar="<32>")
    parser.add_argument("--ro", type=int, help="Number of rows per subarray", default=1024, metavar="<1024>")
    parser.add_argument("--co", type=int, help="Number of columns per subarray", default=8192, metavar="<8192>")
    parser.add_argument("--mem-config", type=str, help="Path to memory configuration file", metavar="<file>")
    parser.add_argument("--sim-config", type=str, help="Path to simulator configuration file (sim_target will be ignored)", metavar="<file>")
    parser.add_argument("--load-balance", type=int, help="Load balancing", default=None, choices=[0, 1])
    parser.add_argument("--max-threads", type=int, help="Maximum number of threads", default=None, metavar="<int>")
    parser.add_argument("--debug", type=int, help="Debug flags", default=None, metavar="<int>")

    # Show help if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Parse known arguments
    args, unknown_args = parser.parse_known_args()

    # Device ID must be valid
    if 0 <= args.sim_target < len(device_names):
        device_name = device_names[args.sim_target]
    else:
        print("Error: Valid device IDs are 0 to", str(len(device_names) - 1))
        print(device_help)
        sys.exit(1)

    # DRAM dimension needs to be a power of 2
    def is_power_of_2(n):
        return (n & (n - 1) == 0) and n != 0
    if not is_power_of_2(args.ra):
        print("Error: num ranks must be a power of 2")
        sys.exit(1)
    if not is_power_of_2(args.ba):
        print("Error: num banks must be a power of 2")
        sys.exit(1)
    if not is_power_of_2(args.sa):
        print("Error: num subarrays must be a power of 2")
        sys.exit(1)
    if not is_power_of_2(args.ro):
        print("Error: num rows must be a power of 2")
        sys.exit(1)
    if not is_power_of_2(args.co):
        print("Error: num cols must be a power of 2")
        sys.exit(1)

    # Print the configuration
    print("*** PIMeval App Launcher ***")
    print("Launching PIMeval application with parameters:")
    print("  Simulation Target:", device_name)
    print("  Number of Ranks:", args.ra)
    print("  Number of Banks:", args.ba)
    print("  Number of Subarrays:", args.sa)
    print("  Number of Rows:", args.ro)
    print("  Number of Columns:", args.co)

    # Set environment variables for simulation
    os.environ["PIMEVAL_SIM_TARGET"] = device_name
    os.environ["PIMEVAL_NUM_RANKS"] = str(args.ra)
    os.environ["PIMEVAL_NUM_BANK_PER_RANK"] = str(args.ba)
    os.environ["PIMEVAL_NUM_SUBARRAY_PER_BANK"] = str(args.sa)
    os.environ["PIMEVAL_NUM_ROW_PER_SUBARRAY"] = str(args.ro)
    os.environ["PIMEVAL_NUM_COL_PER_SUBARRAY"] = str(args.co)
    if (args.mem_config is not None):
        # Get absolute path of the memory configuration file
        mem_config_abs_path = os.path.abspath(args.mem_config)
        print("  Memory Config:", mem_config_abs_path)
        os.environ["PIMEVAL_MEM_CONFIG"] = mem_config_abs_path
    if (args.sim_config is not None):
        # Get absolute path of the simulator configuration file
        sim_config_abs_path = os.path.abspath(args.sim_config)
        print("  Simulator Config:", sim_config_abs_path, "(May overwrite above settings)")
        os.environ["PIMEVAL_SIM_CONFIG"] = sim_config_abs_path
    if (args.load_balance is not None):
        print("  Load Balance:", args.load_balance)
        os.environ["PIMEVAL_LOAD_BALANCE"] = str(args.load_balance)
    if (args.max_threads is not None):
        print("  Max Threads:", args.max_threads)
        os.environ["PIMEVAL_MAX_NUM_THREADS"] = str(args.max_threads)
    if (args.debug is not None):
        print("  Debug:", args.debug)
        os.environ["PIMEVAL_DEBUG"] = str(args.debug)

    # Execute the application
    print("Running executable:", args.executable)
    print("----------------------------------------")
    try:
        # Ensure the executable has a valid prefix if it's a relative path
        executable_path = args.executable
        if not os.path.isabs(executable_path) and not executable_path.startswith("./"):
            executable_path = "./" + executable_path
        os.execvp(executable_path, [executable_path] + unknown_args)
    except OSError as e:
        print("Execution failed:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()

