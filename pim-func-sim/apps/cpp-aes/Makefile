# Makefile for AES
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

SUBDIRS := PIM

.PHONY: debug perf dramsim3_integ clean $(SUBDIRS)
.DEFAULT_GOAL := perf

debug perf dramsim3_integ clean: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

