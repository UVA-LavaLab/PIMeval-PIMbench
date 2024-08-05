# Makefile for PIMeval Simulator - Applications
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

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

