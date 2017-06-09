#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include "caesar.h"
#include "util.h"


int main(int argc, char *argv[])
{
    //printf("ROT-13 tests: %s\n", rot13_test() ? "SUCCEEDED" : "FAILED");
    BYTE *data;
    struct stat st;

    if (argc != 3) {
        printf("Uso: ./caesar nome_arquivo c\n");
        exit(EXIT_FAILURE);
    }

    data = read_file(argv[1]);
    caesar(argv[2], data);
    free(data);
    return(0);
}