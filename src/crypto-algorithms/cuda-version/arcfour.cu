//standard includes
#include <stdio.h>
#include <stdlib.h>

//strcmp
#include <string.h>

//stat struct
#include <sys/stat.h>

//memory comparation
#include <memory.h>

#include "arcfour.cuh"

extern "C" {
//read file and size functions
    #include "util.h"
}

#define N (2048 * 2048)
#define NUM_THREADS 512

__device__ void arcfour_key_setup(BYTE state[], BYTE key[], int len)
{
    int i, j;
    BYTE t;

    for (i = 0; i < 256; ++i)
        state[i] = i;
    for (i = 0, j = 0; i < 256; ++i) {
        j = (j + state[i] + key[i % len]) % 256;
        t = state[i];
        state[i] = state[j];
        state[j] = t;
    }
}

__device__ void arcfour_generate_stream(BYTE state[], BYTE out[], size_t len)
{
    int i, j;
    size_t idx;
    BYTE t;

    for (idx = 0, i = 0, j = 0; idx < len; ++idx)  {
        i = (i + 1) % 256;
        j = (j + state[i]) % 256;
        t = state[i];
        state[i] = state[j];
        state[j] = t;
        out[idx] = state[(state[i] + state[j]) % 256];
    }
}

__global__ void generate_key(BYTE* generated_key) 
{	
    BYTE state[256];
    BYTE key[3][10] = {{"Key"}, {"Wiki"}, {"Secret"}};
    int idx = 0; 

    for (idx = 0; idx < 3; idx++)
      arcfour_key_setup(state, key[idx], 3);
	printf("state: %s", state); 
  
    arcfour_generate_stream(state, generated_key, 256);
}

__global__ void xor_encrypt(BYTE* data, BYTE* key, int* len) 
{
    int i;

    for(i = 0; i < *len; i++) {
      data[i] = data[i] ^ key[i % (sizeof(key)/sizeof(char))];
    }
}

void print_error_message(cudaError_t err, const char *var, int type) {
    if (err != cudaSuccess) {
        if (type == ALLOC) {
            fprintf(stderr, "Falha na alocacao de %s\n", var);
        } else {
            fprintf(stderr, "Falha na copia de %s\n", var);
        }
        exit(EXIT_FAILURE);
    }
}

void enc_file(char *filename, char *enc_filename) 
{
    cudaError_t err; 
    printf("enc_file \n");
    // BYTE *data;
    // BYTE *enc_data;
    BYTE generated_key[1024];
    // size_t len;
    // BYTE *d_data = NULL;
    // BYTE *d_key = NULL;
    // int *d_len = NULL;
    // cudaError_t err = cudaSuccess;
    
    // data = read_file(filename);
    // len = get_file_size();
    generate_key<<<N/NUM_THREADS, NUM_THREADS>>>(generated_key);
    fprintf(stderr, "%s \n", generated_key);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
          fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(err));
          exit(EXIT_FAILURE);
    }

    // enc_data = (BYTE *) malloc(len * sizeof(BYTE));

    // err = cudaMalloc(&d_data, len * sizeof(BYTE));
    // print_error_message(err, (const char *) "d_data", ALLOC); 

    // err = cudaMalloc(&d_len, sizeof(int));
    // print_error_message(err, (const char *) "d_len", ALLOC);

    // err = cudaMalloc(&d_key, 1024 * sizeof(BYTE));
    // print_error_message(err, (const char *) "d_key", ALLOC); 

    // err = cudaMemcpy(d_data, data, len * sizeof(BYTE), cudaMemcpyHostToDevice);
    // print_error_message(err, (const char *) "d_data", COPY);

    // err = cudaMemcpy(d_len, &len, sizeof(int), cudaMemcpyHostToDevice);
    // print_error_message(err, (const char *) "d_len", COPY);

    // err = cudaMemcpy(d_key, (const void *) 1024, sizeof(BYTE), cudaMemcpyHostToDevice);
    // print_error_message(err, (const char *) "d_key", COPY);
 
    // xor_encrypt <<<N/NUM_THREADS, NUM_THREADS>>>(d_data, d_key, d_len);
    
    // err = cudaMemcpy(enc_data, d_data, len * sizeof(BYTE), cudaMemcpyDeviceToHost);
    // print_error_message(err, (const char *) "enc_data", COPY);

    // FILE *enc_file = fopen(enc_filename, "wb");
    // fwrite(enc_data, len * sizeof(BYTE), 1, enc_file); 
    // free(enc_data);
    // cudaFree(d_data);
    // cudaFree(d_len);
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Uso: ./arcfour modo\n");
        printf("Para encriptar um arquivo: ./arcfour -e nome_arquivo arquivo_encriptado\n");
        printf("Para rodar os testes: ./arcfour -t\n");
        exit(EXIT_FAILURE);
    }
	
    if (strcmp(argv[1], "-e") == 0) {
	enc_file(argv[2], argv[3]);
    } 

    return(0);
}
