/*********************************************************************
* FILEname:   rot-13_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding ROT-13
              implementation. These tests do not encompass the full
              range of available test vectors, however, if the tests
              pass it is very, very likely that the code is correct
              and was compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include "rot-13.h"
#include "util.c"

/*********************** FUNCTION DEFINITIONS ***********************/
int rot13_test()
{
    char text[] = {"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"};
    char code[] = {"NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"};
    char buf[1024];
    int pass = 1;

    // To encode, just apply ROT-13.
    strcpy(buf, text);
    rot13(buf);
    pass = pass && !strcmp(code, buf);

    // To decode, just re-apply ROT-13.
    rot13(buf);
    pass = pass && !strcmp(text, buf);

    return(pass);
}

void dec_file(char *enc_filename, BYTE *data) {
    BYTE *enc_data;
    struct stat st;
    enc_data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);
    strcpy(enc_data, data);
    rot13(enc_data);
    FILE *enc_file = fopen(enc_filename, "wb+");
    fwrite(enc_data, sizeof(BYTE) * st.st_size, 1, enc_file);
    free(enc_data);
    fclose(enc_file);
}

int main(int argc, char *argv[])
{
    //printf("ROT-13 tests: %s\n", rot13_test() ? "SUCCEEDED" : "FAILED");
    BYTE *data;

    struct stat st;
    if (argc != 3) {
        printf("uso: \n");
        printf("./rot-13 nome_arquivo nome_arquivo_criptografado\n");
        exit(EXIT_FAILURE);
    }

    data = read_file(argv[1]);
    dec_file(argv[2], data);
    free(data);
    return(0);
}
