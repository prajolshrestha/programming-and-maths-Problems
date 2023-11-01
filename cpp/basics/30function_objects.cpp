#include <iostream>
#include <vector>
#include <algorithm>

struct IsDivisible {
    int divisor;
    IsDivisible(int new_divisor) : divisor(new_divisor) {}

    // function call operator ==> creates function object
    bool operator()(int dividend) {
        return dividend % divisor == 0;
    }
};

int main() {
    IsDivisible is_divisibleby_10(10); // initialize

    // Task 1:
    std::cout<<is_divisibleby_10(50)<<'\n'; // call object (like functions)

    // Task 2: c++20 using std::find_if()
    std::vector<int> my_vector = {41, 20, 31,44,532};
    auto itr = std::ranges::find_if(my_vector, is_divisibleby_10);
    std::cout<< *itr <<'\n';

    return 0;
}