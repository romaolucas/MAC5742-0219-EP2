OBJECTS= md5.o util.o
OUTPUT=md5
NVCC=nvcc
LINK_FLAGS=--device-c

all: build

build: $(OBJECTS)
	$(NVCC) -o $(OUTPUT) $^

md5.o: md5.cu md5.cuh
	$(NVCC) $(LINK_FLAGS) $^

util.o: util.c util.h
	$(NVV) $(LINK_FLAGS) $^


clean:
	rm $(OBJECTS) $(OUTPUT)
