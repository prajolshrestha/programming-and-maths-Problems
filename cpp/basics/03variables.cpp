#include <iostream>

int main() {
    // Lets define integer variable
    int var1;
    var1 = 10; // assigning value to our variable

    // Define double precision floating-point variable and assign value in same line. 
    double var2 = 20.5; // Good practice

    // automatic type deduction of addition operation
    auto var3 = var1 + var2; 

    std::cout<< var3 << '\n';

    return 0;
}