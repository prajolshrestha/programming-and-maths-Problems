// An example program with an out of bound index
#include <array>
#include <iostream>

int main() {
    std::array<int, 5> my_array = {42, 3, 39, 4, 9};

    // a for loop with an off-by-one error
    for (auto i = 0u; i <= my_array.size(); i++) { // accidently we did '<='
        std::cout << my_array[i] << '\n';
    }
}

// ./0_error.exe        
//42 3 39 4 9 0  // last ma grabage value cha