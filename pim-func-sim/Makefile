# Makefile for PIM Functional Simulator
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

LIBDIR := libpimsim
APPDIR := apps
TESTDIR := tests

.PHONY: debug perf dramsim3_integ clean $(LIBDIR) $(APPDIR) $(TESTDIR)
.DEFAULT_GOAL := perf

debug: $(LIBDIR) $(APPDIR) $(TESTDIR)
	@echo "\nINFO: Built PIM Functional Simulator with target = debug\n"

perf: $(LIBDIR) $(APPDIR) $(TESTDIR)
	@echo "\nINFO: Built PIM Functional Simulator with target = perf\n"

dramsim3_integ: $(LIBDIR) $(APPDIR) $(TESTDIR)
	@echo "\nINFO: Built PIM Functional Simulator with target = dramsim3_integ\n"

clean: $(LIBDIR) $(APPDIR) $(TESTDIR)

# Run make with PIM_SIM_TARGET=<PimDeviceEnum> to override default simulation target
PIM_SIM_TARGET ?= PIM_DEVICE_NONE

# Run make with USE_OPENMP=1 to enable OpenMP in some apps
USE_OPENMP ?= 0

$(LIBDIR) $(APPDIR) $(TESTDIR):
	$(MAKE) -C $@ $(MAKECMDGOALS) PIM_SIM_TARGET=$(PIM_SIM_TARGET) USE_OPENMP=$(USE_OPENMP)

