#include "caesar.h"

void caesar(int c, char str[])
{
	int l = strlen(str);
	const char *alpha[2] = { "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"};
	printf("%s", str)
 
	int i;
	for (i = 0; i < l; i++)
	{
		if (!isalpha(str[i]))
			continue;
 
		str[i] = alpha[isupper(str[i])][((int)(tolower(str[i])-'a')+c)%26];
	}
}