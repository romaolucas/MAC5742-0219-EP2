OBJECTS_ROT13=rot-13.o util.o
OBJECTS_DES=des.o util.o
OBJECTS_ARCFOUR=arcfour.o util.o
NVCC=nvcc
LINK_FLAGS=--device-c -G -g -arch=sm_50

all: rot-13 des arcfour

rot-13: $(OBJECTS_ROT13)
	$(NVCC) -arch=sm_50 -o $@ $^

des: $(OBJECTS_DES)
	$(NVCC) -arch=sm_50 -o $@ $^

arcfour: $(OBJECTS_ARCFOUR)
	$(NVCC) -arch=sm_50 -o $@ $^

util.o: util.c 
	$(NVCC) $(LINK_FLAGS) $^

rot-13.o: rot-13.cu util.o
	$(NVCC) $(LINK_FLAGS) $^

des.o: des.cu util.o
	$(NVCC) $(LINK_FLAGS) $^

arcfour.o: arcfour.cu util.o
	$(NVCC) $(LINK_FLAGS) $^

clean:
	rm *.o rot-13 des arcfour