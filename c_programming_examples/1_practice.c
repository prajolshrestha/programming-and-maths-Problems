#include <stdio.h> //library which include function such as
#include <stdlib.h>

// main function
int main(){
    int lucky = 23;//type name value


    printf("value: %i \n", lucky); // print value
    printf("value: %p \n", &lucky); // print address in memory

    //char hello; //char type ==> represents 1 byte character stored as an integer
    //char hello[] = "Hi Friend!" //string is array of char
    char *str = malloc(4);// make a string and allocate 4 byte

    str[0] = 'h';
    str[1] = 'e';
    str[2] = 'y';
    str[3] = '\0';

    // all done now so release memory
    free(str);


    return 0; //returns int type where 0 = success & 1 = failure
}