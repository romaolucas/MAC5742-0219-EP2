#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/stat.h>
#include <string.h>
#include "util.h"

BYTE * read_file(char *filename) {
	BYTE *data;
    struct stat st;

    if (stat(filename, &st) == 0) {
        data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);
    }

    FILE *file = fopen(filename, "rb");
    file_size = st.st_size;
    if (data != NULL && file) {
        int current_byte = 0;
        while (fread(&data[current_byte], sizeof(BYTE), 1, file) == 1) {
            current_byte++;        
        }
    }

    fclose(file);
    return data;
}

void write_file(char* filename, BYTE* data) {
    struct stat st;
    FILE *file = fopen(filename, "wb+");
    fwrite(data, sizeof(BYTE) * st.st_size, 1, file);
    fclose(file);
}

void print_hex(BYTE *vec, int vec_len) {
    int i;
    for (i = 0; i < vec_len; i++) {
        printf("%02x", vec[i]);
    }
    printf("\n");
}

size_t get_file_size() {
    return file_size;
}
