#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/stat.h>
#include <string.h>
#define BYTE unsigned char


BYTE * read_file(char *filename) {
	BYTE *data;
    struct stat st;

    if (stat(filename, &st) == 0) {
        data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);
    }

    FILE *file = fopen(filename, "rb");

    if (data != NULL && file) {
        int current_byte = 0;
        while (fread(&data[current_byte], sizeof(BYTE), 1, file) == 1) {
            current_byte++;        
        }
    }

    fclose(file);
    return data;
}
