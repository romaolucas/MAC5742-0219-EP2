CC=gcc
CC_OPT=-g

.PHONY: all
all: arcfour md5 rot-13 sha1

util.o: util.c util.h
	$(CC) -c $(CC_OPT) $^

arcfour: arcfour_test.c arcfour.c arcfour.h util.o
	$(CC) -o $@ $(CC_OPT) $^

md5: md5_test.c md5.c md5.h util.o
	$(CC) -o $@ $(CC_OPT) $^

rot-13: rot-13_test.c rot-13.c rot-13.h util.o
	$(CC) -o $@ $(CC_OPT) $^

sha1: sha1_test.c sha1.c sha1.h util.o
	$(CC) -o $@ $(CC_OPT) $^

.PHONY: clean
clean:
	rm arcfour md5 rot-13 sha1
