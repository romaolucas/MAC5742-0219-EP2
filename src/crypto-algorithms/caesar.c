#include "caesar.h"

void caesar(int c, char str[])
{
	int i;
	int l = strlen(str);
	const char *alpha[2] = { "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"};
 
	for (i = 0; i < l; i++)
	{
		printf("oi %i\n", l);
		if (!isalpha(str[i]))
			continue;
 
		str[i] = alpha[isupper(str[i])][((int)(tolower(str[i])-'a')+c)%26];
	}
}