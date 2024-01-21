#include <iostream>

int main() {
    // Define and assign value to Variables 
    int a = 15; // "=" sign for assignment
    int b = 1;

    // conditional statement (chained)
    if (a < b) {
        std::cout<< "a is less than b!\n";
    } else if (a == b){ // "==" sign for equility
        std::cout<< " a is equal to b!\n";
    } else {
        std::cout<< "a is greater than b!\n";
        if (a == 15) {
            std::cout<<"a is equal to 15!\n";
        }
    }

    return 0; 
}