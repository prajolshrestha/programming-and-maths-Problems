#include <iostream>
#include <array>

// Create two different versions of same function using templates
template<typename T>
void print_array1(T array){
    for (auto value : array) {
        std::cout<<value<<' ';
    }
    std::cout<<'\n';
}

// templated functing using auto  // g++ 08function_templates.cpp -o templates --std=c++20
void print_array(auto array){
    for (auto value : array) {
        std::cout<<value<<' ';
    }
    std::cout<<'\n';
}

int main() {

    std::array<int, 3> my_int_array = {1,2,3};
    std::array<float, 3> my_float_array = {1.1f, 2.2f, 3.3f};

    // Type aware function call
    print_array<std::array<int, 3>>(my_int_array);
    print_array<std::array<float, 3>>(my_float_array);

    // You can also call function normally (similar to automatic type deduction). Compilar will figure it out.
    print_array(my_int_array);
    print_array(my_float_array);

    return 0;
}