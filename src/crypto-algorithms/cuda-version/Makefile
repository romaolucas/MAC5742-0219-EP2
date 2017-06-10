OBJECTS= arcfour.o util.o
OUTPUT=arcfour
NVCC=nvcc
LINK_FLAGS=--device-c -G -g -arch=sm_50

all: build

build: $(OBJECTS)
	$(NVCC) -arch=sm_50 -o $(OUTPUT) $^

util.o: util.c 
	$(NVCC) $(LINK_FLAGS) $^

arcfour.o: arcfour.cu util.o
	$(NVCC) $(LINK_FLAGS) $^

clean:
	rm $(OBJECTS) $(OUTPUT)
