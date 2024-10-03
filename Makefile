# Makefile for PIMeval / PIMbench Framework
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

LIBDIR := libpimeval
BITSERIALDIR := bit-serial
APPDIR := PIMbench
TESTDIR := misc-bench tests
ALLDIRS := $(LIBDIR) $(BITSERIALDIR) $(APPDIR) $(TESTDIR)

# Handle dependency between lib and apps to support make -j
DEP_LIBPIMEVAL := $(LIBDIR)/lib/libpimeval.a

.PHONY: debug perf dramsim3_integ clean $(ALLDIRS)
.DEFAULT_GOAL := perf

debug: $(ALLDIRS)
	@echo "\nINFO: Built PIMeval Simulator with target = debug\n"

perf: $(ALLDIRS)
	@echo "\nINFO: Built PIMeval Simulator with target = perf\n"

dramsim3_integ: $(ALLDIRS)
	@echo "\nINFO: Built PIMeval Simulator with target = dramsim3_integ\n"

clean: $(ALLDIRS)

# Run make with PIM_SIM_TARGET=<PimDeviceEnum> to override default simulation target
PIM_SIM_TARGET ?= PIM_DEVICE_NONE

# Run make with USE_OPENMP=1 to enable OpenMP in some apps
USE_OPENMP ?= 0

# Run make with COMPILE_WITH_JPEG=0 to disable compilation with JPEG. JPEG compilation is needed for VGG apps.
COMPILE_WITH_JPEG ?= 0

$(DEP_LIBPIMEVAL) $(LIBDIR):
	$(MAKE) -C $(LIBDIR) $(MAKECMDGOALS) PIM_SIM_TARGET=$(PIM_SIM_TARGET) USE_OPENMP=$(USE_OPENMP) COMPILE_WITH_JPEG=$(COMPILE_WITH_JPEG)

$(BITSERIALDIR) $(APPDIR) $(TESTDIR): $(DEP_LIBPIMEVAL)
	$(MAKE) -C $@ $(MAKECMDGOALS) PIM_SIM_TARGET=$(PIM_SIM_TARGET) USE_OPENMP=$(USE_OPENMP) COMPILE_WITH_JPEG=$(COMPILE_WITH_JPEG)

