// Automatic formatting
#include <iostream>
#include <array>
#include <algorithm>

int main() {
    std::array<int, 5>
        my_array = 
        {1, 2, 3, 4, 5};
    std::sort(my_array.begin(),
        my_array.end()
        );

    for (auto value : my_array) { std::cout<<value <<' ';}
    std::cout<<'\n';
        return 
        0;
}