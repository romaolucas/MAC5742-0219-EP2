//standard includes
#include <stdio.h>
#include <stdlib.h>

//stat struct
#include <sys/stat.h>


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

int main(int argc, char *argv[])
{
    cudaError_t err = cudaSuccess;
    BYTE *data;
    BYTE *d_data = NULL;
    BYTE *enc_data;
    size_t len;
    size_t *d_len = NULL;

    if (argc != 3) {
        printf("Uso: ./rot-13 nome_arquivo nome_arquivo_criptografado\n");
        exit(EXIT_FAILURE);
    }

    data = read_file(argv[1]);
    len = get_file_size();
    enc_data = (BYTE *) malloc(len * sizeof(BYTE));

    err = cudaMalloc(&d_data, len * sizeof(BYTE));
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha na alocacao do d_data\n");
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&d_len, sizeof(size_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha na alocacao do d_len\n");
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_data, data, len * sizeof(BYTE), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha na copia do data para o device\n");
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_len, &len, sizeof(size_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha na copia do len para o device\n");
        exit(EXIT_FAILURE);
    }


    rot13 <<<N/NUM_THREADS, NUM_THREADS>>>(d_data, d_len);
    
    err = cudaMemcpy(enc_data, d_data, len * sizeof(BYTE), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Falha na copia do data para o host\n");
        exit(EXIT_FAILURE);
    }

    FILE *enc_file = fopen(argv[2], "wb");
    fwrite(enc_data, len * sizeof(BYTE), 1, enc_file);
    free(enc_data);
    fclose(enc_file);
    free(data);
    cudaFree(d_data);
    cudaFree(d_len);
    return(0);
}
