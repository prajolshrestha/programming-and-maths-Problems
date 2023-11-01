#include <iostream>

struct A {
    A() {
        std::cout<< "Constructing A!\n";
    }
};

// solution of diamond problem: make A virtual class
struct B : virtual A {
    B() {
        std::cout<< "Constructing B!\n";
    }
};

struct C : virtual A {
    C() {
        std::cout<< "Constructing C!\n";
    }
};

// Diamond Problem: D is inheriting A twice. (once through B and once through C)
struct D : B, C{ 
    D() {
        std::cout<< "Constructing D!\n";
    }
};

int main() {
    D d;

    A &a = d; // we can't do upcasting in polymorphysm! // solution: with virtual class A, we can do it!
    return 0;
}