#ifndef ARCFOUR_CUH
#define ARCFOUR_CUH

#define ALLOC 0 
#define COPY 1

#define TRUE 1
#define FALSE 0

typedef unsigned char BYTE;

void arcfour_key_setup(BYTE state[], const BYTE key[], int len);

__global__ void arcfour_generate_stream(BYTE state[], BYTE out[], size_t len);


__device__ BYTE generate_key();

__global__ void xor_encrypt(BYTE* data, BYTE* key, int len);

void print_error_message(cudaError_t err, const char *var, int type);

void enc_file(char *filename, char *enc_filename);

#endif
