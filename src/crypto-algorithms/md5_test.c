/*********************************************************************
* Filename:   md5_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding MD5
              implementation. These tests do not encompass the full
              range of available test vectors, however, if the tests
              pass it is very, very likely that the code is correct
              and was compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <memory.h>
#include <string.h>
#include "md5.h"

/*********************** FUNCTION DEFINITIONS ***********************/
int md5_test()
{
    BYTE text1[] = {""};
    BYTE text2[] = {"abc"};
    BYTE text3_1[] = {"ABCDEFGHIJKLMNOPQRSTUVWXYZabcde"};
    BYTE text3_2[] = {"fghijklmnopqrstuvwxyz0123456789"};
    BYTE hash1[MD5_BLOCK_SIZE] = {0xd4,0x1d,0x8c,0xd9,0x8f,0x00,0xb2,0x04,0xe9,0x80,0x09,0x98,0xec,0xf8,0x42,0x7e};
    BYTE hash2[MD5_BLOCK_SIZE] = {0x90,0x01,0x50,0x98,0x3c,0xd2,0x4f,0xb0,0xd6,0x96,0x3f,0x7d,0x28,0xe1,0x7f,0x72};
    BYTE hash3[MD5_BLOCK_SIZE] = {0xd1,0x74,0xab,0x98,0xd2,0x77,0xd9,0xf5,0xa5,0x61,0x1c,0x2c,0x9f,0x41,0x9d,0x9f};
    BYTE buf[16];
    MD5_CTX ctx;
    int pass = 1;

    md5_init(&ctx);
    md5_update(&ctx, text1, strlen(text1));
    md5_final(&ctx, buf);
    pass = pass && !memcmp(hash1, buf, MD5_BLOCK_SIZE);

    // Note the MD5 object can be reused.
    md5_init(&ctx);
    md5_update(&ctx, text2, strlen(text2));
    md5_final(&ctx, buf);
    pass = pass && !memcmp(hash2, buf, MD5_BLOCK_SIZE);

    // Note the data is being added in two chunks.
    md5_init(&ctx);
    md5_update(&ctx, text3_1, strlen(text3_1));
    md5_update(&ctx, text3_2, strlen(text3_2));
    md5_final(&ctx, buf);
    pass = pass && !memcmp(hash3, buf, MD5_BLOCK_SIZE);

    return(pass);
}

void md5_file(char *filename) {
    BYTE *data;
    BYTE hash[MD5_BLOCK_SIZE];
    MD5_CTX ctx;

    struct stat st;

    if (stat(filename, &st) == 0) {
        data = (BYTE *) malloc(sizeof(BYTE) * st.st_size);
    }

    FILE *file = fopen(filename, "rb");

    if (data != NULL && file) {
        int current_byte = 0;
        while (fread(&data[current_byte], sizeof(BYTE), 1, file) == 1) {
            current_byte++;        
        }
    }

    md5_init(&ctx);
    md5_update(&ctx, data, strlen(data));
    md5_final(&ctx, hash);
    int i;
    for (i = 0; i < MD5_BLOCK_SIZE; i++) {
        printf("%02x", hash[i]);
    }
    printf("\n");
    fclose(file);
    free(data);
}

int main()
{
    //printf("MD5 tests: %s\n", md5_test() ? "SUCCEEDED" : "FAILED");
    if (argc != 2) {
        printf("uso: ./md5 nome-arquivo\n");
        exit(EXIT_FAILURE);
    }
    md5_file(argv[1]);

    return(0);
}
