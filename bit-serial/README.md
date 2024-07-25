## Bit-serial Micro-program Evaluation Framework

### Description

This code module contains detailed bit-serial micro-program implementations of all PIMeval high-level functional APIs and different data types for each bit-serial PIM architecture. This demonstrates how PIMeval simulator supports low-level micro-ops programming using architecture specific micro-ops.

For simulation speed, we don't run these bit-serial micro-programs for each API call during functional simulation, i.e., the `PIM_FUNCTIONAL` device used by PIMbench suite. Instead, we run this code module offline, collect detailed stats, and embed the results as part of the PIMeval performance and energy modeling.

### How to Run

```
make -j<n_proc>
./bitSerial.out
```

### Code Organization

* `bitSerialMain`: Main entry to run all bit-serial micro-programs
* `bitSerialBase`: Base interface class with common code to verify the correctness of micro-programs
* `bitSerial<arch>`: Detailed bit-serial micro-program implementations for a bit-serial PIM architecture

