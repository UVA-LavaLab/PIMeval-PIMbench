# Makefile for PIM Functional Simulator - Applications
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

SUBDIRS := $(wildcard */.)

.PHONY: debug perf dramsim3_integ clean $(SUBDIRS)
.DEFAULT_GOAL := perf

USE_OPENMP ?= 0

COMPILE_WITH_JPEG ?= 0

debug: $(SUBDIRS)
	@echo "INFO: apps target = debug"

perf: $(SUBDIRS)
	@echo "INFO: apps target = perf"

dramsim3_integ: $(SUBDIRS)
	@echo "INFO: apps target = dramsim3_integ"

clean: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS) USE_OPENMP=$(USE_OPENMP) COMPILE_WITH_JPEG=$(COMPILE_WITH_JPEG)

