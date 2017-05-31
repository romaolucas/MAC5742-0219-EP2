#include <iostream>
#include <string>
#include "util.c"

__global__ void rot13(BYTE* str, int len)
{
   int case_type, idx;
   for (idx = 0; idx < len; idx++) {
      // Only process alphabetic characters.
      if (str[idx] < 'A' || (str[idx] > 'Z' && str[idx] < 'a') || str[idx] > 'z')
         continue;
      // Determine if the char is upper or lower case.
      if (str[idx] >= 'a')
         case_type = 'a';
      else
         case_type = 'A';
      // Rotate the char's value, ensuring it doesn't accidentally "fall off" the end.
      str[idx] = (str[idx] + 13) % (case_type + 26);
      if (str[idx] < 26)
         str[idx] += case_type;
   }
}

int main(int argc, char *argv[])
{
    //printf("ROT-13 tests: %s\n", rot13_test() ? "SUCCEEDED" : "FAILED");
    BYTE *data;
    struct stat st;

    if (argc != 3) {
        printf("Uso: ./rot-13 nome_arquivo nome_arquivo_criptografado\n");
        exit(EXIT_FAILURE);
    }

    data = read_file(argv[1]);
    rot13 <<<1, 1>>>(data, sizeof(BYTE) * st.st_size);
    free(data);
    return(0);
}