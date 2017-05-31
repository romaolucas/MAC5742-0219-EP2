#include <iostream>
#include <string>
#define CIPHER_NUMBER 13

using namespace std;

__global__ void rot13(char* str, int len)
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

int main()
{
    char text[] = "Hello world!";
    char* d_text;
    cudaMalloc(&d_text, 12*sizeof(char));
    cudaMemcpy(d_text, text, 12*sizeof(char), cudaMemcpyHostToDevice);
    rot13 <<<1, 1>>>(d_text, 12);
    cudaMemcpy(text, d_text, 12*sizeof(char), cudaMemcpyDeviceToHost);
    cout << "The answer is: " << text << endl;
    cudaFree(d_text);
    return 0;
}