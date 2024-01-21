// g++ main.cpp -o main  ==> doesn't work because our compilar cant find the header file 'print.h'

// Lets create object file for main.cpp and print.cpp and then link them together.
//Compile main file: g++ main.cpp -o main.o -c
//Compile print file(our file which is included in header file print.h): g++ print.cpp -o print.o -c 
//Link two compiled file to get exe file: g++ main.o print.o -o main

#include "print.h"

//void print(int a); // just define , there is some print function somewhere // but we have to link it. so it doesnt work. so we create print.h file

int main() {
    print(10);
    return 0;
}