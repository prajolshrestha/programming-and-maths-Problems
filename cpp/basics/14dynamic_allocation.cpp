#include <iostream>

int main() {
    // dynamically allocate space of a single integer
    int *int_ptr = new int;  //allocate
    *int_ptr = 242;
    std::cout<< "Value: " << *int_ptr <<'\n'; // dereference garne
    std::cout<< "Address: " << int_ptr <<'\n'; 

    delete int_ptr; // Free memory

    ///////////////////////////////////////////////////////////////

    // Dynamically allocate space for array of 10 integer
    int *int_ptr1 = new int[10]; // dynamically allocate 
    int_ptr1[0] = 42;
    std::cout<< "Value: " << int_ptr1[0] <<'\n'; // dereference garne
    std::cout<< "Address: " << &int_ptr1[0] <<'\n'; 

    delete[] int_ptr1; // Free memory

    return 0;
}