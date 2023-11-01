#include <iostream>
#include <vector>
#include <utility>

struct S {
    // Constructor
    S() {
        std::cout<<"Constructor!\n";
    }

    // Copy Construtor, so there will be no "move constructor"!
    S(const S &s) {
        std::cout<< "Copy constructor!\n";
    }

    // Move constructor(takes rvalue)
    S(S &&s) {
        std::cout<<"Move Constructor!\n";
    }
};

int main() {
    std::vector<S> my_vector;
    S s; // It calls constructor
    //my_vector.push_back(s); // It calls copy constructor.
    //my_vector.push_back(std::move(s)); // It calls move constructor.

    my_vector.push_back(S()); // we gave rvalue so it calls move constructor

    return 0;
}