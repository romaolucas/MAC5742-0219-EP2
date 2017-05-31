
CC=gcc
CC_OPT=-g

.PHONY: all
all: arcfour rot-13 sha1

arcfour: arcfour_test.c arcfour.c arcfour.h
	$(CC) -o $@ $(CC_OPT) $^

rot-13: rot-13_test.c rot-13.c rot-13.h
	$(CC) -o $@ $(CC_OPT) $^

sha1: sha1_test.c sha1.c sha1.h
	$(CC) -o $@ $(CC_OPT) $^

.PHONY: clean
clean:
	rm $(OUTPUT)
