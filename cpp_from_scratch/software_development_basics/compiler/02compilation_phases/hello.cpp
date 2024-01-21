// Pre-processing phase: g++ hello.cpp -o hello.ii -E
// Compilation Phase: g++ hello.ii -o hello.s -S
// Assembly : g++ hello.s -o hello.o -c 
// Linking: g++ hello.o -o hello

#include <iostream>

int main() {
    std::cout<<"Hello, World!\n";
    return 0;
}