OBJECTS= md5.o util.o
OUTPUT=md5
NVCC=nvcc
LINK_FLAGS=--device-c -G -g

all: build

build: $(OBJECTS)
	$(NVCC) -g -G -o $(OUTPUT) $^

md5.o: md5.cu
	$(NVCC) $(LINK_FLAGS) $^

util.o: util.c
	$(NVCC) $(LINK_FLAGS) $^


clean:
	rm $(OBJECTS) $(OUTPUT)
