#include <iostream>
#include <memory>

int main() {

    // Method 1:
    std::shared_ptr<int[]> ptr1(new int[10]);
    auto ptr2 = ptr1;

    std::cout<< "Reference count: "<< ptr1.use_count();
    std::cout<<'\n';

    // Method 2: //  g++ 16shared_ptr.cpp -o shared_ptr --std=c++20
    auto ptr3 = std::make_shared<int[]>(10);
    auto ptr4 = ptr3;
    std::cout<< "Reference count: "<< ptr3.use_count();


    return 0; 
}