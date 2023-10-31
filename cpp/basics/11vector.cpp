#include <iostream>
#include <vector>

void print(std::vector<int> vector) {
    for( auto value : vector) {
        std::cout<< value << ' ';
    }
    std::cout<< '\n';
}

// vector can dynamically grow
int main() {
    // Task 1:
    std::vector<int> my_vector = {1,2,3,4,5};
    print(my_vector);
    
    // Modifiers
    my_vector.push_back(6);
    print(my_vector);

    my_vector.pop_back();
    print(my_vector);

    // Task 2:
    std::vector<int> my_vector1;
    my_vector1.reserve(10); // reserves storage

    for (int i = 0; i < 10; i++) {
        std::cout<< "size: " << my_vector1.size() << '\n'; // size : number of elements inside a vector
        std::cout<< "Capacity: " << my_vector1.capacity() << '\n'; // Capacity: number of elements that can be held in currently allocated storage

        my_vector1.push_back(i);
    }


    return 0;
}