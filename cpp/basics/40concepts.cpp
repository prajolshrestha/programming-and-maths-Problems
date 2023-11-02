#include <iostream>
#include <concepts>


void print(std::integral auto value) { // we defined integral type in out function template
    std::cout<<"Print integral value: "<< value <<'\n';
}

int main() {
    print(10); // int 8bit, 16bit, 32bit, 64bit
    //print(10.352); // double float can't be used
    return 0;
}