/*********************************************************************
* Filename:   rot-13_test.c
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
#include <unistd.h>
#include "rot-13.h"

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

void enc_dec_file(char *filename) {
    char *data;
    char *enc_data;

    struct stat st;

    if (stat(filename, &st) == 0) {
        data = (char *) malloc(sizeof(char) * st.size);
        enc_data = (char *) malloc(sizeof(char) * st.size);
    }

    File *file = fopen(filename, "rb");

    if (data != NULL && file) {
        int current_byte = 0;
        while (fread(&data[current_byte], sizeof(char), 1, file) == 1) {
            current_byte++;
        }
    }
    strcpy(data, enc_data);
    rot13(enc_data);

    File *enc_file = fopen("moby_dick_enc.txt", "wb+");

    fwrite(enc_data, sizeof(char) * st.size, 1, enc_file);

    fclose(file);
    fclose(enc_data);
}

int main()
{
    printf("ROT-13 tests: %s\n", rot13_test() ? "SUCCEEDED" : "FAILED");

    return(0);
}
