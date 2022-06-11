# Makefile for 'bag' module
#
# CS50, Summer 2021

PROG = bagtest
OBJS = bagtest.o bag.o ../lib/readlinep.o ../lib/memory.o 
LIBS =

# uncomment the following to turn on verbose memory logging
#TESTING=-DMEMTEST

CFLAGS = -Wall -pedantic -std=c11 -ggdb $(TESTING) -I../lib
CC = gcc
MAKE = make

$(PROG): $(OBJS)
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

bagtest.o: bag.h ../lib/readlinep.h
bag.o: bag.h
readlinep.o: ../lib/readlinep.h

.PHONY: test valgrind clean

# expects a file `test.names` to exist; it can contain any text.
test: $(PROG) test.names
	./$(PROG) < test.names

# expects a file `test.names` to exist; it can contain any text.
valgrind: $(PROG) test.names
	valgrind ./$(PROG) < test.names

clean:
	rm -rf *.dSYM  # MacOS debugger info
	rm -f *~ *.o
	rm -f $(PROG)

