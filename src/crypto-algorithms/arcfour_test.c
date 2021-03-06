/*********************************************************************
* Filename:   arcfour_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding ARCFOUR
              implementation. These tests do not encompass the full
              range of available test vectors, however, if the tests
              pass it is very, very likely that the code is correct
              and was compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include "arcfour.h"
#include "util.h"

/*********************** FUNCTION DEFINITIONS ***********************/
int rc4_test()
{
    BYTE state[256];
    BYTE key[3][10] = {{"Key"}, {"Wiki"}, {"Secret"}};
    BYTE stream[3][10] = {{0xEB,0x9F,0x77,0x81,0xB7,0x34,0xCA,0x72,0xA7,0x19},
                          {0x60,0x44,0xdb,0x6d,0x41,0xb7},
                          {0x04,0xd4,0x6b,0x05,0x3c,0xa8,0x7b,0x59}};
    int stream_len[3] = {10,6,8};
    BYTE buf[1024];
    int idx;
    int pass = 1;

    // Only test the output stream. Note that the state can be reused.
    for (idx = 0; idx < 3; idx++) {
        arcfour_key_setup(state, key[idx], strlen(key[idx]));
        arcfour_generate_stream(state, buf, stream_len[idx]);
        pass = pass && !memcmp(stream[idx], buf, stream_len[idx]);
    }

    return(pass);
}

BYTE *xor_encrypt(BYTE* input, BYTE* key, size_t len) 
{
    BYTE* output = (BYTE *) malloc(sizeof(BYTE) * len);;
    int i;

    for(i = 0; i < len; i++) {
      output[i] = input[i] ^ key[i % (sizeof(key)/sizeof(char))];
    }

    return output;
}


int main(int argc, char *argv[])
{
    //printf("ARCFOUR tests: %s\n", rc4_test() ? "SUCCEEDED" : "FAILED");
    BYTE *data, *output;
    BYTE generated_key[1024];
    BYTE state[256];
    BYTE key[3][10] = {{"Key"}, {"Wiki"}, {"Secret"}};
    int stream_len[3] = {10,6,8};
    int idx;
    size_t len;
    struct stat st;
    if (argc != 3) {
        printf("Uso: ./arcfour nome_arquivo nome_arquivo_criptografado\n");
        printf("Ou ./arcfour -tf nome_arquivo\n");
        exit(EXIT_FAILURE);
    }
    
    for (idx = 0; idx < 3; idx++)
        arcfour_key_setup(state, key[idx], strlen(key[idx]));
    
    arcfour_generate_stream(state, generated_key, strlen(state));
    
    if (strcmp(argv[1], "-tf") == 0) {
        data = read_file(argv[2]);
        len = get_file_size();
        output = xor_encrypt(data, generated_key, len);
        output = xor_encrypt(output, generated_key, len);
        assert(!memcmp(data, output, len * sizeof(BYTE)));
    } else {
        data = read_file(argv[1]);
        len = get_file_size();
        output = xor_encrypt(data, generated_key, len);
        write_file(argv[2], output);
    }
    
    free(data);
    free(output);
    
    return(0);
}
