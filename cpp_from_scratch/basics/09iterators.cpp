#include <array>
#include <iostream>

int main() {
    std::array<int, 5> my_array = {1,2,3,4,5};
    // for loop with its bounds initialized as iterators
    for (auto itr = my_array.begin(); itr < my_array.end(); itr += 1) {
        std::cout<< *itr << ' ';
    }
    std::cout << '\n';

    // iterate over range but with some offset (subset of container)
    for (auto itr = my_array.begin() + 2; itr < my_array.end(); itr += 1) {
        std::cout<< *itr << ' ';
    }
    std::cout << '\n';

    // Reverse printing is so easy with iterators
    for (auto itr = my_array.rbegin(); itr < my_array.rend(); itr += 1) {
        std::cout<< *itr << ' ';
    }
    std::cout << '\n';



    return 0;
}