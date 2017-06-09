#include "caesar.h"

void caesar(int c, char str[])
{
	int i, aux;
	int l = strlen(str);
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