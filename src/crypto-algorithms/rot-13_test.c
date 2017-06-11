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
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <memory.h>
#include <assert.h>
#include "rot-13.h"
#include "util.h"

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
    size_t len;
    len = get_file_size();
    enc_data = (BYTE *) malloc(len * sizeof(BYTE));
    memcpy(enc_data, data, len * sizeof(BYTE));
    rot13(enc_data);
    write_file(enc_filename, enc_data);
    free(enc_data);
}

void enc_dec_file(char *filename) {
    BYTE *data;
    BYTE *enc_data;
    BYTE *dec_data;
    size_t len;

    data = read_file(filename);
    len = get_file_size();

    enc_data = (BYTE *) malloc(len * sizeof(BYTE));
    memcpy(enc_data, data, len * sizeof(BYTE));

    rot13(enc_data);

    dec_data = (BYTE *) malloc(len * sizeof(BYTE));
    memcpy(dec_data, enc_data, len * sizeof(BYTE));

    rot13(dec_data);
    assert(!memcmp(data, dec_data, len * sizeof(BYTE)));

    free(data);
    free(enc_data);
    free(dec_data);
}

int main(int argc, char *argv[])
{
    BYTE *data;

    struct stat st;
    if (argc < 2) {
        printf("Uso: ./rot-13 nome_arquivo nome_arquivo_criptografado\n");
        printf("Ou ./rot-13 -tf nome_arquivo\n");
        exit(EXIT_FAILURE);
    }

    if (strcmp(argv[1], "-tf") == 0) {
        enc_dec_file(argv[2]);
    } else {
        data = read_file(argv[1]);
        dec_file(argv[2], data);
        free(data);
    }
    return(0);
}
