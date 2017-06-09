OBJECTS= rot-13.o util.o
OUTPUT=rot-13
NVCC=nvcc
LINK_FLAGS=--device-c -G -g -arch=sm_50

all: build

build: $(OBJECTS)
	$(NVCC) -arch=sm_50 -o $(OUTPUT) $^

util.o: util.c 
	$(NVCC) $(LINK_FLAGS) $^

rot-13.o: rot-13.cu util.o
	$(NVCC) $(LINK_FLAGS) $^

clean:
	rm $(OBJECTS) $(OUTPUT)
