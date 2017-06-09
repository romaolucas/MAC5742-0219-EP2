#ifndef CAESAR_H
#define CAESAR_H

/*************************** HEADER FILES ***************************/
#include <stdio.h>
#include <string.h>
#include <ctype.h>

// #define caesar(x) rot(13, x)
// #define decaesar(x) rot(13, x)
// #define decrypt_rot(x, y) rot((26-x), y)

/*********************** FUNCTION DECLARATIONS **********************/
// Performs IN PLACE rotation of the input. Assumes input is NULL terminated.
// Preserves each charcter's case. Ignores non alphabetic characters.
void caesar(int c, char str[]);

#endif 
