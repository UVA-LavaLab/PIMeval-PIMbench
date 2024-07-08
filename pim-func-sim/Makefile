# Makefile for PIM Functional Simulator
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

LIBDIR := libpimsim
APPDIR := apps
TESTDIR := tests
# Handle dependency between lib and apps to support make -j
DEP_LIBPIMSIM := $(LIBDIR)/lib/libpimsim.a

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

# Run make with COMPILE_WITH_JPEG=0 to disable compilation with JPEG. JPEG compilation is needed for VGG apps.
COMPILE_WITH_JPEG ?= 1 

$(DEP_LIBPIMSIM) $(LIBDIR):
	$(MAKE) -C $(LIBDIR) $(MAKECMDGOALS) PIM_SIM_TARGET=$(PIM_SIM_TARGET) USE_OPENMP=$(USE_OPENMP) COMPILE_WITH_JPEG=$(COMPILE_WITH_JPEG)

$(APPDIR) $(TESTDIR): $(DEP_LIBPIMSIM)
	$(MAKE) -C $@ $(MAKECMDGOALS) PIM_SIM_TARGET=$(PIM_SIM_TARGET) USE_OPENMP=$(USE_OPENMP) COMPILE_WITH_JPEG=$(COMPILE_WITH_JPEG)

