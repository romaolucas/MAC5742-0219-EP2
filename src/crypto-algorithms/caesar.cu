#include <iostream>
#include <string.h>
#include "util.c"

__global__ void caesar(int c, BYTE* str, int l)
{
  int i, aux;
  const char *alpha[2] = { "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"};
 
  for (i = 0; i < l; i++)
  {
    if (!isalpha(str[i]))
      continue;

    if (isupper(str[i]))
      aux = 1;
    else
      aux = 0;
 
    str[i] = alpha[aux][((int)(tolower(str[i])-'a')+c)%26];
  }
}

int main(int argc, char *argv[])
{
    BYTE *data;
    struct stat st;

    if (argc != 2) {
        printf("Uso: ./caesar nome_arquivo\n");
        exit(EXIT_FAILURE);
    }

    data = read_file(argv[1]);
    caesar <<<1, 1>>>(20, data, sizeof(BYTE) * st.st_size);
    free(data);
    return(0);
}