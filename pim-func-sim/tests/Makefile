# Makefile for PIM Functional Simulator - Tests
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

SUBDIRS := $(wildcard */.)

.PHONY: debug perf dramsim3_integ clean $(SUBDIRS)
.DEFAULT_GOAL := perf

debug: $(SUBDIRS)
	@echo "INFO: tests target = debug"

perf: $(SUBDIRS)
	@echo "INFO: tests target = perf"

dramsim3_integ: $(SUBDIRS)
	@echo "INFO: tests target = dramsim3_integ"

clean: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

