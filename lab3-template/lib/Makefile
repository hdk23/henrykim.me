# Makefile for 'lib' module
#
# CS50

# PROG =
OBJS = memory.o readlinep.o
LIBS =

# uncomment the following to turn on verbose memory logging
# (comment or uncomment the following, then "make clean; make")
#TESTING=-DMEMTEST

CFLAGS = -Wall -pedantic -std=c11 -ggdb $(TESTING)
MAKE = make
CC = gcc

default: $(OBJS)

memory.o: memory.h

readlinep.o: readlinep.h

.PHONY: clean

clean:
	rm -rf *.dSYM  # MacOS debugger info
	rm -f *~ *.o
