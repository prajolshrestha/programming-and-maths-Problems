#include <array>
#include <iostream>
#include <algorithm> // includes sort

void print(auto array) {
    for(auto value : array){
        std::cout<< value <<' ';
    }
    std::cout<< '\n';
}

int main() {
    std::array<int, 5> my_array = {53, 21, 3, 4, 98};

    print(my_array);

    // method 1:
    //std::sort(my_array.begin(), my_array.end()); // we can also add offset if we want to sort subset of container.

    // method 2: (easy)
    std::ranges::sort(my_array);

    print(my_array);

    
    return 0;
}