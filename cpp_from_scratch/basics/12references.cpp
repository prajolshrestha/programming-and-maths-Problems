#include <iostream>

int main() {

    // Straight forward way (Expensive, copying is required)
    int a = 5;
    int b = a; // we created new integer to store value of a
    b += 1;

    std::cout<< "a = "<< a << '\n';
    std::cout<< "b = "<< b << '\n';
    // lets inspect the address of these different variables. Where are they in memory?
    std::cout<< "a = "<< &a << '\n';
    std::cout<< "b = "<< &b << '\n';

    /////////////////////////////////////////////////////////////////////////////
    // Instead of creating new integer, lets create reference to an integer
    // I want 'b' to reference 'a', instead of asking for a new piece of memory
    // Reference: lets give another name that refers to sth.(variable, obj, fun, ..) that already exists.

    int c = 5;
    int &d = c; // we are not asking a piece of memory anymore to store a new integer b here
    d += 1;

    std::cout<< "c = "<< c << '\n';
    std::cout<< "d = "<< d << '\n';

    // lets inspect the address of these different variables. Where are they in memory?
    std::cout<< "c = "<< &c << '\n';
    std::cout<< "d = "<< &d << '\n';

    // Conclusion: 'd' is an alias to already-existing object or function.
                   // Both c and d have exact same memory address.

    return 0;
}