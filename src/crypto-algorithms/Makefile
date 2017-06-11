CC=gcc
CC_OPT=-g

.PHONY: all
all: arcfour des rot-13

util.o: util.c util.h
	$(CC) -c $(CC_OPT) $^

arcfour: arcfour_test.c arcfour.c arcfour.h util.o
	$(CC) -o $@ $(CC_OPT) $^

des: des_test.c des.c des.h util.o
	$(CC) -o $@ $(CC_OPT) $^

rot-13: rot-13_test.c rot-13.c rot-13.h util.o
	$(CC) -o $@ $(CC_OPT) $^

.PHONY: clean
clean:
	rm arcfour md5 rot-13 sha1
