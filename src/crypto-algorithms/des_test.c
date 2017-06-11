/*********************************************************************
* Filename:   des_test.c
* Author:     Brad Conte (brad AT bradconte.com)
* Copyright:
* Disclaimer: This code is presented "as is" without any guarantees.
* Details:    Performs known-answer tests on the corresponding DES
              implementation. These tests do not encompass the full
              range of available test vectors, however, if the tests
              pass it is very, very likely that the code is correct
              and was compiled properly. This code also serves as
              example usage of the functions.
*********************************************************************/

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "des.h"
#include "util.h"

/*********************** FUNCTION DEFINITIONS ***********************/
int des_test()
{
    BYTE pt1[DES_BLOCK_SIZE] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xE7};
    BYTE pt2[DES_BLOCK_SIZE] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF};
    BYTE pt3[DES_BLOCK_SIZE] = {0x54,0x68,0x65,0x20,0x71,0x75,0x66,0x63};
    BYTE ct1[DES_BLOCK_SIZE] = {0xc9,0x57,0x44,0x25,0x6a,0x5e,0xd3,0x1d};
    BYTE ct2[DES_BLOCK_SIZE] = {0x85,0xe8,0x13,0x54,0x0f,0x0a,0xb4,0x05};
    BYTE ct3[DES_BLOCK_SIZE] = {0xc9,0x57,0x44,0x25,0x6a,0x5e,0xd3,0x1d};
    BYTE ct4[DES_BLOCK_SIZE] = {0xA8,0x26,0xFD,0x8C,0xE5,0x3B,0x85,0x5F};
    BYTE key1[DES_BLOCK_SIZE] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF};
    BYTE key2[DES_BLOCK_SIZE] = {0x13,0x34,0x57,0x79,0x9B,0xBC,0xDF,0xF1};
    BYTE three_key1[DES_BLOCK_SIZE * 3] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
                                           0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
                                           0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF};
    BYTE three_key2[DES_BLOCK_SIZE * 3] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
                                           0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,0x01,
                                           0x45,0x67,0x89,0xAB,0xCD,0xEF,0x01,0x23};

    BYTE schedule[16][6];
    BYTE three_schedule[3][16][6];
    BYTE buf[DES_BLOCK_SIZE];
    int pass = 1;

    des_key_setup(key1, schedule, DES_ENCRYPT);
    des_crypt(pt1, buf, schedule);
    pass = pass && !memcmp(ct1, buf, DES_BLOCK_SIZE);

    des_key_setup(key1, schedule, DES_DECRYPT);
    des_crypt(ct1, buf, schedule);
    pass = pass && !memcmp(pt1, buf, DES_BLOCK_SIZE);

    des_key_setup(key2, schedule, DES_ENCRYPT);
    des_crypt(pt2, buf, schedule);
    pass = pass && !memcmp(ct2, buf, DES_BLOCK_SIZE);

    des_key_setup(key2, schedule, DES_DECRYPT);
    des_crypt(ct2, buf, schedule);
    pass = pass && !memcmp(pt2, buf, DES_BLOCK_SIZE);

    three_des_key_setup(three_key1, three_schedule, DES_ENCRYPT);
    three_des_crypt(pt1, buf, three_schedule);
    pass = pass && !memcmp(ct3, buf, DES_BLOCK_SIZE);

    three_des_key_setup(three_key1, three_schedule, DES_DECRYPT);
    three_des_crypt(ct3, buf, three_schedule);
    pass = pass && !memcmp(pt1, buf, DES_BLOCK_SIZE);

    three_des_key_setup(three_key2, three_schedule, DES_ENCRYPT);
    three_des_crypt(pt3, buf, three_schedule);
    pass = pass && !memcmp(ct4, buf, DES_BLOCK_SIZE);

    three_des_key_setup(three_key2, three_schedule, DES_DECRYPT);
    three_des_crypt(ct4, buf, three_schedule);
    pass = pass && !memcmp(pt3, buf, DES_BLOCK_SIZE);

    return(pass);
}

void enc_dec_file(char *filename)
{
    BYTE *data;
    BYTE *encrypted_data;
    BYTE *decrypted_data;

    BYTE key1[DES_BLOCK_SIZE] = {0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF};
    BYTE schedule[16][6];
    size_t len;

    data = read_file(filename);
    len = get_file_size();

    encrypted_data = (BYTE *) malloc(sizeof(BYTE) * len);
    decrypted_data = (BYTE *) malloc(sizeof(BYTE) * len);

    BYTE data_buf[DES_BLOCK_SIZE];
    BYTE data_enc[DES_BLOCK_SIZE];
    BYTE data_dec[DES_BLOCK_SIZE];

    for(int i = 0; i < len; i++){
        for(int j = 0; j < DES_BLOCK_SIZE; j++){
            if(i < len){
                data_buf[j] = data[i];
                i++;
            };
        };

        des_key_setup(key1, schedule, DES_ENCRYPT);
        des_crypt(data_buf, data_enc, schedule);

        des_key_setup(key1, schedule, DES_DECRYPT);
        des_crypt(data_enc, data_dec, schedule);

        i -= DES_BLOCK_SIZE;
        for(int k = 0; k < DES_BLOCK_SIZE; k++){
            if(i < len){
                encrypted_data[i] = data_enc[k];
                decrypted_data[i] = data_dec[k];
                i++;
            };
        };

        i--;
    };

    free(decrypted_data);
    free(encrypted_data);
    free(data);
};

int main(int argc, char *argv[])
{
    if (argc == 3) {
        enc_dec_file(argv[2]);
    } else {
        printf("DES test: %s\n", des_test() ? "SUCCEEDED" : "FAILED");
    }
    return(0);
}
