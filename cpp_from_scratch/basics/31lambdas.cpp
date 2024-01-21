#include <iostream>
#include <vector>
#include <algorithm>

// Completely replace IsDivisible struct!!!

int main() {
    // Lambda expression
    // initialize as unkonwn function object
    // [captures ﻿]  (params ﻿)  { body }
    auto is_divisibleby_10 = [divisor = 10](int dividend) {
        return dividend % divisor == 0;
    }; 

    // Task 1:
    std::cout<<is_divisibleby_10(50)<<'\n'; // call object (like functions)

    // Task 2: c++20 using std::find_if()
    std::vector<int> my_vector = {41, 20, 31,44,532};
    auto itr = std::ranges::find_if(my_vector, is_divisibleby_10);
    std::cout<< *itr <<'\n';

    return 0;
}