#include <iostream>

#include "singleton.hpp"

int main(){
    // ILLEGAL! Constructor is private!
    // Singleton *test = new Singleton();

    // Creates a new singleton, and returns a pointer to it.
    Singleton *simple_singleton_1 = Singleton::getInstance();

    // only returns a pointer to the single singleton instance
    Singleton *simple_singleton_2 = Singleton::getInstance();

    std::cout<< "Singleton 1: "<< simple_singleton_1<< "Singleton 2: "<<simple_singleton_2 << std::endl;

    return 0;
}

// Compile , link obj file and create exe
// g++ main.cpp singleton.cpp -o main
