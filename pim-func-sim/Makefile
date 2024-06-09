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

# Run "make PIM_SIM_TARGET=<PimDeviceEnum>" to override default simulation target
PIM_SIM_TARGET ?= PIM_DEVICE_NONE

$(LIBDIR) $(APPDIR) $(TESTDIR):
	$(MAKE) -C $@ $(MAKECMDGOALS) PIM_SIM_TARGET=$(PIM_SIM_TARGET)

