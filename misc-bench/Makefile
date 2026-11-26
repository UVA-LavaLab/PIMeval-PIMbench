<<<<<<< HEAD
# Makefile for PIMeval Simulator - Applications
=======
<<<<<<< HEAD
# Makefile for PIMeval / PIMbench Framework
=======
# Makefile for PIMeval Simulator - Applications
>>>>>>> 556bf46 (host_PIM_Prefixsum)
>>>>>>> 14b1d2b (host_PIM_Prefixsum)
# Copyright (c) 2024 University of Virginia
# This file is licensed under the MIT License.
# See the LICENSE file in the root of this repository for more details.

<<<<<<< HEAD
SUBDIRS := $(wildcard */.)
=======
<<<<<<< HEAD
LIBDIR := libpimeval
BITSERIALDIR := bit-serial
APPDIR := PIMbench
TESTDIR := misc-bench tests
ALLDIRS := $(LIBDIR) $(BITSERIALDIR) $(APPDIR) $(TESTDIR)
>>>>>>> 14b1d2b (host_PIM_Prefixsum)

.PHONY: debug perf dramsim3_integ clean $(SUBDIRS)
.DEFAULT_GOAL := perf

USE_OPENMP ?= 0

COMPILE_WITH_JPEG ?= 0

debug: $(SUBDIRS)
	@echo "INFO: apps target = debug"

<<<<<<< HEAD
=======
$(BITSERIALDIR) $(APPDIR) $(TESTDIR): $(DEP_LIBPIMEVAL)
	$(MAKE) -C $@ $(MAKECMDGOALS) PIM_SIM_TARGET=$(PIM_SIM_TARGET) USE_OPENMP=$(USE_OPENMP) COMPILE_WITH_JPEG=$(COMPILE_WITH_JPEG)
=======
SUBDIRS := $(wildcard */.)

.PHONY: debug perf dramsim3_integ clean $(SUBDIRS)
.DEFAULT_GOAL := perf

USE_OPENMP ?= 0

COMPILE_WITH_JPEG ?= 0

debug: $(SUBDIRS)
	@echo "INFO: apps target = debug"

>>>>>>> 14b1d2b (host_PIM_Prefixsum)
perf: $(SUBDIRS)
	@echo "INFO: apps target = perf"

dramsim3_integ: $(SUBDIRS)
	@echo "INFO: apps target = dramsim3_integ"

clean: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS) USE_OPENMP=$(USE_OPENMP) COMPILE_WITH_JPEG=$(COMPILE_WITH_JPEG)
<<<<<<< HEAD
=======
>>>>>>> 556bf46 (host_PIM_Prefixsum)
>>>>>>> 14b1d2b (host_PIM_Prefixsum)

