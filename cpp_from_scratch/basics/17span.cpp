#include <iostream>
#include <span>
#include <vector>

// we are not copying 'my_vector' to 'span', we are just looking at 'my_vector' without owning the underlying memory
void print_subvector(std::span<int> span) {
    for (auto value : span) {
        std::cout<< value <<' ';
    }
    std::cout<<'\n';
}

int main() {
    std::vector<int> my_vector = {1,2,3,4,5};
    print_subvector(my_vector);
    print_subvector(std::span(my_vector.begin(), 2)); // printing subset of vector
    print_subvector(std::span(my_vector.begin() + 1, 3)); // printing subset of vector

    return 0;
}