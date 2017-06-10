#ifndef DES_CUH
#define DES_CUH

#define ALLOC 0 
#define COPY 1

#define TRUE 1
#define FALSE 0

#define DES_BLOCK_SIZE 8

typedef unsigned char BYTE; 
typedef unsigned int  WORD;

typedef enum {
    DES_ENCRYPT,
    DES_DECRYPT
} DES_MODE;


__device__ void des_key_setup(const BYTE key[], BYTE schedule[][6], DES_MODE mode);

__device__ void des_crypt(const BYTE in[], BYTE out[], const BYTE key[][6]);

__global__ void des(BYTE *data, BYTE *data_enc, BYTE *data_dec, size_t *len);

int test_des();

void enc_file(char *filename, char *enc_filename);

void print_error_message(cudaError_t err, const char *var, int type);

#endif
