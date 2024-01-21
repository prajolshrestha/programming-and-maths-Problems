#include <iostream>
#include <array>

int main() {
    // Array
    std::array<int, 5> my_array = {42, 12, 63, 1, 3};

    // for loop
    for (int value : my_array){
        std::cout<<value<<'\n';
    }

    return 0;
}