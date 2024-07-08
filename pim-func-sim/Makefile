# Makefile for PIM Functional Simulator
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

LIBDIR := libpimsim
BITSERIALDIR := bit-serial
APPDIR := apps
TESTDIR := tests
ALLDIRS := $(LIBDIR) $(BITSERIALDIR) $(APPDIR) $(TESTDIR)

# Handle dependency between lib and apps to support make -j
DEP_LIBPIMSIM := $(LIBDIR)/lib/libpimsim.a

.PHONY: debug perf dramsim3_integ clean $(ALLDIRS)
.DEFAULT_GOAL := perf

debug: $(ALLDIRS)
	@echo "\nINFO: Built PIM Functional Simulator with target = debug\n"

perf: $(ALLDIRS)
	@echo "\nINFO: Built PIM Functional Simulator with target = perf\n"

dramsim3_integ: $(ALLDIRS)
	@echo "\nINFO: Built PIM Functional Simulator with target = dramsim3_integ\n"

clean: $(ALLDIRS)

# Run make with PIM_SIM_TARGET=<PimDeviceEnum> to override default simulation target
PIM_SIM_TARGET ?= PIM_DEVICE_NONE

# Run make with USE_OPENMP=1 to enable OpenMP in some apps
USE_OPENMP ?= 0

$(DEP_LIBPIMSIM) $(LIBDIR):
	$(MAKE) -C $(LIBDIR) $(MAKECMDGOALS) PIM_SIM_TARGET=$(PIM_SIM_TARGET) USE_OPENMP=$(USE_OPENMP)

$(BITSERIALDIR) $(APPDIR) $(TESTDIR): $(DEP_LIBPIMSIM)
	$(MAKE) -C $@ $(MAKECMDGOALS) PIM_SIM_TARGET=$(PIM_SIM_TARGET) USE_OPENMP=$(USE_OPENMP)

