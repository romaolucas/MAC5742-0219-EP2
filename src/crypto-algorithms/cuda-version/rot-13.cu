//standard includes
#include <stdio.h>
#include <stdlib.h>

//assert func
#include <assert.h>

//strcmp
#include <string.h>

//stat struct
#include <sys/stat.h>

//memory comparation
#include <memory.h>

#include "rot-13.cuh"

extern "C" {
//read file and size functions
    #include "util.h"
}

#define N (2048 * 2048)
#define NUM_THREADS 512



__global__ void rot13(BYTE* str, size_t *len)
{
    int case_type, idx;
    idx = threadIdx.x + blockDim.x * blockIdx.x;
    case_type = 'A';
    if (idx < *len) {
        if (str[idx] < 'A' || (str[idx] > 'Z' && str[idx] < 'a') || str[idx] > 'z') {
            return;
         } else {
            if (str[idx] >= 'a') {
                case_type = 'a';
            }
            str[idx] = (str[idx] + 13) % (case_type + 26);
            if (str[idx] < 26) {
                str[idx] += case_type;
            }
        }
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

int test_rot_13() {
    cudaError_t err = cudaSuccess;
    BYTE *data;
    BYTE *d_data = NULL;
    BYTE *dec_data;
    size_t len;
    size_t *d_len = NULL;
    const char *samples[3] = {"../sample_files/moby_dick.txt", "../sample_files/hubble_1.tif", "../sample_files/mercury.png"};
    int passed = TRUE;
    int i;
    for (i = 0; i < 3; i++) {
        printf("Testando arquivo %s\n", samples[i]);
        data = read_file((char *) samples[i]);
        len = get_file_size();
        dec_data = (BYTE *) malloc(len * sizeof(BYTE));

        err = cudaMalloc(&d_data, len * sizeof(BYTE));
        print_error_message(err, (const char *) "d_data", ALLOC); 

        err = cudaMalloc(&d_len, sizeof(size_t));
        print_error_message(err, (const char *) "d_len", ALLOC);

        err = cudaMemcpy(d_data, data, len * sizeof(BYTE), cudaMemcpyHostToDevice);
        print_error_message(err, (const char *) "d_data", COPY);

        err = cudaMemcpy(d_len, &len, sizeof(size_t), cudaMemcpyHostToDevice);
        print_error_message(err, (const char *) "d_len", COPY);
     
        rot13 <<<N/NUM_THREADS, NUM_THREADS>>>(d_data, d_len);
        rot13 <<<N/NUM_THREADS, NUM_THREADS>>>(d_data, d_len);

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

void enc_file(char *filename, char *enc_filename) {
    BYTE *data;
    BYTE *enc_data;
    size_t len;
    BYTE *d_data = NULL;
    size_t *d_len = NULL;
    cudaError_t err = cudaSuccess;
    
    data = read_file(filename);
    len = get_file_size();
    enc_data = (BYTE *) malloc(len * sizeof(BYTE));

    err = cudaMalloc(&d_data, len * sizeof(BYTE));
    print_error_message(err, (const char *) "d_data", ALLOC); 

    err = cudaMalloc(&d_len, sizeof(size_t));
    print_error_message(err, (const char *) "d_len", ALLOC);

    err = cudaMemcpy(d_data, data, len * sizeof(BYTE), cudaMemcpyHostToDevice);
    print_error_message(err, (const char *) "d_data", COPY);

    err = cudaMemcpy(d_len, &len, sizeof(size_t), cudaMemcpyHostToDevice);
    print_error_message(err, (const char *) "d_len", COPY);
 
    rot13 <<<N/NUM_THREADS, NUM_THREADS>>>(d_data, d_len);
    
    err = cudaMemcpy(enc_data, d_data, len * sizeof(BYTE), cudaMemcpyDeviceToHost);
    print_error_message(err, (const char *) "enc_data", COPY);

    FILE *enc_file = fopen(enc_filename, "wb");
    fwrite(enc_data, len * sizeof(BYTE), 1, enc_file); 
    fclose(enc_file);
    free(enc_data);
    cudaFree(d_data);
    cudaFree(d_len);
}

void enc_dec_file_test(char *filename) {
    BYTE *data;
    BYTE *dec_data;
    size_t len;
    BYTE *d_data = NULL;
    size_t *d_len = NULL;
    cudaError_t err = cudaSuccess;
    
    data = read_file(filename);
    len = get_file_size();
    dec_data = (BYTE *) malloc(len * sizeof(BYTE));

    err = cudaMalloc(&d_data, len * sizeof(BYTE));
    print_error_message(err, (const char *) "d_data", ALLOC); 

    err = cudaMalloc(&d_len, sizeof(size_t));
    print_error_message(err, (const char *) "d_len", ALLOC);

    err = cudaMemcpy(d_data, data, len * sizeof(BYTE), cudaMemcpyHostToDevice);
    print_error_message(err, (const char *) "d_data", COPY);

    err = cudaMemcpy(d_len, &len, sizeof(size_t), cudaMemcpyHostToDevice);
    print_error_message(err, (const char *) "d_len", COPY);
 
    rot13 <<<N/NUM_THREADS, NUM_THREADS>>>(d_data, d_len);
    rot13 <<<N/NUM_THREADS, NUM_THREADS>>>(d_data, d_len);

    err = cudaMemcpy(dec_data, d_data, len * sizeof(BYTE), cudaMemcpyDeviceToHost);
        print_error_message(err, (const char *) "dec_data", COPY);
    
    assert(!memcmp(data, dec_data, len * sizeof(BYTE)));
    free(dec_data);
    cudaFree(d_data);
    cudaFree(d_len);
}

void show_usage() {
    printf("Uso: \n");
    printf("Para encriptar um arquivo: ./rot-13 -e nome_arquivo arquivo_encriptado\n");
    printf("Para rodar os testes: ./rot-13 -t\n");
    printf("Para testar decriptacao de um arquvio: ./rot-13 -tf nome_arquivo\n");
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        show_usage();
        exit(EXIT_FAILURE);
    }

    if (strcmp(argv[1], "-e") == 0) {
        if (argc != 4) {
            show_usage();
            exit(EXIT_FAILURE);
        }
        enc_file(argv[2], argv[3]);
    } else if (strcmp(argv[1], "-t") == 0) {
        int passed = test_rot_13();
        if (passed == TRUE) {
            printf("Todos os testes passaram!\n");
        }
    } else if (strcmp(argv[1], "-tf") == 0) {
        if (argc != 3) {
            show_usage();
            exit(EXIT_FAILURE);
        }
        enc_dec_file_test(argv[2]);
    }else {
        printf("Opção inválida\n");
        show_usage();
        exit(EXIT_FAILURE);
    }

    return(0);
}
