OUTPUT=md5

CC=gcc
CC_OPT=-g

.PHONY: all
all: $(OUTPUT)

$(OUTPUT): $(OUTPUT)_test.c $(OUTPUT).c $(OUTPUT).h
	$(CC) -o $(OUTPUT) $(CC_OPT) $(OUTPUT)_test.c $(OUTPUT).c $(OUTPUT).h

.PHONY: clean
clean:
	rm $(OUTPUT)
