#include <iostream>

// Raw pointer
int main() {
    int a = 5;
    int *b = &a; // b is a pointer pointing towards integer and stores address of integer variable.
    *b += 1; //dereferencing pointer and accessing value of 'a' through 'b' and modifying it

    std::cout<<"a = "<< a << '\n';
    std::cout<<"address of a = "<< &a << '\n';
    std::cout<<"b = "<< b << '\n'; //b is just a variable storing address

    return 0;

}