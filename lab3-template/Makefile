# Makefile for CS50 Lab3 
#   Builds and compiles all the data structures.
#
# uncomment the following to turn on verbose memory logging
# (and run `make clean; make` whenever you change this)
# TESTING=-DMEMTEST

# Our Make program and its flags
MAKE = make #TESTING=$(TESTING)

# recursively make in each subdirectory
all:
	$(MAKE) --directory=lib	
	$(MAKE) --directory=bag
	$(MAKE) --directory=counters
	$(MAKE) --directory=set
	$(MAKE) --directory=hashtable


# 'phony' targets are helpful but do not create any file by that name
.PHONY: clean

# to clean up all derived files
clean:
	rm -f *~
	make --directory=lib clean
	make --directory=bag clean
	make --directory=set clean
	make --directory=counters clean
	make --directory=hashtable clean
