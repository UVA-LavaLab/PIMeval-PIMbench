# Makefile for PIM Functional Simulator
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

LIBDIR := libpimsim
APPDIR := apps

.PHONY: debug perf dramsim3_integ clean $(LIBDIR) $(APPDIR)
.DEFAULT_GOAL := perf

debug: $(LIBDIR) $(APPDIR)
	@echo "\nINFO: Built PIM Functional Simulator with target = debug\n"

perf: $(LIBDIR) $(APPDIR)
	@echo "\nINFO: Built PIM Functional Simulator with target = perf\n"

dramsim3_integ: $(LIBDIR) $(APPDIR)
	@echo "\nINFO: Built PIM Functional Simulator with target = dramsim3_integ\n"

clean: $(LIBDIR) $(APPDIR)

$(LIBDIR) $(APPDIR):
	$(MAKE) -C $@ $(MAKECMDGOALS)

