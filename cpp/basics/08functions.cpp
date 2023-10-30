#include <iostream>
#include <array>

// Void return type function 
void print_array(std::array<int, 3> array){
    for (int value : array) {
        std::cout<<value<<' ';
    }
    std::cout<<'\n';
}

// Non void return type function
int sum(std::array<int, 3> array){
    int sum = 0;
    for (int value : array){
        sum += value;
    }
    return sum;
}

// main function
int main() {

    std::array<int, 3> my_array_1 = {1,2,3};
    std::array<int, 3> my_array_2 = {4,5,6};

    // Function call
    print_array(my_array_1);
    print_array(my_array_2);

    int result = sum(my_array_1);

    std::cout<<"sum: " <<result<<'\n';

    return 0;
}