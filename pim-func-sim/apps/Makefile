# Makefile for PIM Functional Simulator - Applications
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

SUBDIRS := $(wildcard */.)

.PHONY: all clean $(SUBDIRS)

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

clean:
	@for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done

