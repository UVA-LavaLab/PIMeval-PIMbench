# Makefile for PIM Functional Simulator
# Copyright 2024 LavaLab @ University of Virginia. All rights reserved.

LIBDIR := libpimsim
APPDIR := apps

.PHONY: all clean $(LIBDIR) $(APPDIR)

all: $(LIBDIR) $(APPDIR)

$(LIBDIR) $(APPDIR):
	$(MAKE) -C $@

clean:
	$(MAKE) -C $(LIBDIR) clean
	$(MAKE) -C $(APPDIR) clean

