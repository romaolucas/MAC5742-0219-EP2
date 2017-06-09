#ifndef ROT13_CUH
#define ROT13_CUH

#define ALLOC 0 
#define COPY 1

#define TRUE 1
#define FALSE 0

typedef unsigned char BYTE;

__global__ void rot13(BYTE *str, size_t *len);

int test_rot_13();

void enc_file(char *filename, char *enc_filename);

void print_error_message(cudaError_t err, const char *var, int type);

#endif
