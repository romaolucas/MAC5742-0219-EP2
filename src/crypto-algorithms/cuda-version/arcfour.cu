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
  
    arcfour_generate_stream(state, generated_key, 256);;
}

__global__ void xor_encrypt(BYTE* data, BYTE* key, int* len) 
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < *len) {
      data[idx] = data[idx] ^ key[idx % (sizeof(key)/sizeof(char))];
    }
}

void print_error_message(cudaError_t err, const char *var, int type) {
    if (err != cudaSuccess) {
        if (type == ALLOC) {
            fprintf(stderr, "Falha na alocacao de %s\n", var);
        } else {
            fprintf(stderr, "Falha na copia de %s\n", var);
        }
        fprintf(stderr, "Erro: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int test_arcfour() {
    cudaError_t err = cudaSuccess;
    BYTE *data;
    BYTE *d_data = NULL;
    BYTE *d_key = NULL;
    BYTE *dec_data;
    BYTE *key;
    size_t len;
    int *d_len = NULL;
    const char *samples[3] = {"../sample_files/moby_dick.txt", "../sample_files/hubble_1.tif", "../sample_files/mercury.png"};
    int passed = TRUE;
    int i;
    for (i = 0; i < 3; i++) {
        printf("Testando arquivo %s\n", samples[i]);
        data = read_file((char *) samples[i]);
        len = get_file_size();
        key = (BYTE *) malloc(1024 * sizeof(BYTE));
        dec_data = (BYTE *) malloc(len * sizeof(BYTE));

        err = cudaMalloc(&d_data, len * sizeof(BYTE));
        print_error_message(err, (const char *) "d_data", ALLOC); 

        err = cudaMalloc(&d_len, sizeof(size_t));
        print_error_message(err, (const char *) "d_len", ALLOC);

        err = cudaMalloc(&d_key, 1024 * sizeof(BYTE));
        print_error_message(err, (const char *) "d_key", ALLOC);

        err = cudaMemcpy(d_data, data, len * sizeof(BYTE), cudaMemcpyHostToDevice);
        print_error_message(err, (const char *) "d_data", COPY);

        err = cudaMemcpy(d_len, &len, sizeof(size_t), cudaMemcpyHostToDevice);
        print_error_message(err, (const char *) "d_len", COPY);

        err = cudaMemcpy(d_key, key, 1024 * sizeof(BYTE), cudaMemcpyHostToDevice);
        print_error_message(err, (const char *) "d_key", COPY);

        generate_key<<<1,1>>>(d_key);
        xor_encrypt <<<N/NUM_THREADS, NUM_THREADS>>>(d_data, d_key, d_len);
        xor_encrypt <<<N/NUM_THREADS, NUM_THREADS>>>(d_data, d_key, d_len);  

        err = cudaMemcpy(dec_data, d_data, len * sizeof(BYTE), cudaMemcpyDeviceToHost);
        print_error_message(err, (const char *) "dec_data", COPY);
    
        passed = passed && !memcmp(data, dec_data, len * sizeof(BYTE));
        if (passed == FALSE) {
            fprintf(stderr, "Problema na decodificacao do %s\n", samples[i]);
        }
        free(dec_data);
        cudaFree(d_data);
        cudaFree(d_len);
    }
    return passed;
}

void enc_file(char *filename, char *enc_filename) 
{
    BYTE *data;
    BYTE *key;
    size_t len;
    BYTE *enc_data;
    BYTE *d_data = NULL;
    BYTE *d_key = NULL;
    int *d_len = NULL;
    cudaError_t err = cudaSuccess;
    
    data = read_file(filename);
    len = get_file_size();;
    
    key = (BYTE *) malloc(1024 * sizeof(BYTE));
    enc_data = (BYTE *) malloc(len * sizeof(BYTE));

    err = cudaMalloc(&d_data, len * sizeof(BYTE));
    print_error_message(err, (const char *) "d_data", ALLOC); 

    err = cudaMalloc(&d_len, sizeof(size_t));
    print_error_message(err, (const char *) "d_len", ALLOC);

    err = cudaMalloc(&d_key, 1024 * sizeof(BYTE));
    print_error_message(err, (const char *) "d_key", ALLOC); 

    err = cudaMemcpy(d_data, data, len * sizeof(BYTE), cudaMemcpyHostToDevice);
    print_error_message(err, (const char *) "d_data", COPY);

    err = cudaMemcpy(d_len, &len, sizeof(size_t), cudaMemcpyHostToDevice);
    print_error_message(err, (const char *) "d_len", COPY);

    err = cudaMemcpy(d_key, key, 1024 * sizeof(BYTE), cudaMemcpyHostToDevice);
    print_error_message(err, (const char *) "d_key", COPY);
 
    generate_key<<<1, 1>>>(d_key);
    
    xor_encrypt <<<N/NUM_THREADS, NUM_THREADS>>>(d_data, d_key, d_len);
    xor_encrypt <<<N/NUM_THREADS, NUM_THREADS>>>(d_data, d_key, d_len);   

    err = cudaMemcpy(enc_data, d_data, len * sizeof(BYTE), cudaMemcpyDeviceToHost);
    print_error_message(err, (const char *) "enc_data", COPY);

    FILE *enc_file = fopen(enc_filename, "wb");
    fwrite(enc_data, len * sizeof(BYTE), 1, enc_file); 
    free(enc_data);
    cudaFree(d_data);
    cudaFree(d_len);
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
    } else {
        int passed = test_arcfour();
        if (passed == TRUE) {
            printf("Todos os testes passaram!\n");
        }
    }

    return(0);
}
