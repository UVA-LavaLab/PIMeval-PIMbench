# Makefile for AES
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

SUBDIRS := PIM

.PHONY: debug perf dramsim3_integ clean $(SUBDIRS)
.DEFAULT_GOAL := perf

USE_OPENMP ?= 0

debug perf dramsim3_integ clean: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS) USE_OPENMP=$(USE_OPENMP)

