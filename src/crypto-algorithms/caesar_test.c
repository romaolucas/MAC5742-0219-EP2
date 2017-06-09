#include <stdlib.h>
#include "caesar.h"
#include "util.h"


int main(int argc, char *argv[])
{
    //printf("ROT-13 tests: %s\n", rot13_test() ? "SUCCEEDED" : "FAILED");
    BYTE *data;

    if (argc != 3) {
        printf("Uso: ./caesar nome_arquivo nome_arquivo_criptografado\n");
        exit(EXIT_FAILURE);
    }

    data = read_file(argv[1]);
    printf("%s", data);
    //caesar(13, data);
    free(data);
    return(0);
}


// int main()
// {
// 	char str[] = "This is a top secret text message!";
 
// 	printf("Original: %s\n", str);
// 	caesar(str);
// 	printf("Encrypted: %s\n", str);
// 	decaesar(str);
// 	printf("Decrypted: %s\n", str);
 
// 	return 0;
// }