#ifndef UTIL_H
#define UTIL_H

#include <stddef.h>

typedef unsigned char BYTE;
size_t file_size;


BYTE * read_file(char *filename);
void write_file(char* filename, BYTE* data);
void print_hex(BYTE *vec, int vec_len);
size_t get_file_size();

#endif //UTIL_H
