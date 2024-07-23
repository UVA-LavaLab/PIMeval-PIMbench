# Makefile for PIMeval Simulator - Tests
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

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

