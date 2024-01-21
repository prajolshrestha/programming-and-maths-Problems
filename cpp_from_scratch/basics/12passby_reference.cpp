#include <iostream>
#include <vector>

// Normal function with pass by value
void add_elements_passby_value(std::vector<int> vector, int N) { // creating completely new integer vector
    for (int i = 0; i < N; i++) {
        vector.push_back(i);
    }
}

// lets create a reference of my_vector # hence pass by reference
void add_elements_using_reference(std::vector<int> &vector, int N) { // vector acts as reference of my_vector (ie, it doesnt copy)
    for (int i = 0; i < N; i++) {
        vector.push_back(i);
    }
}

int main() {
    std::vector<int> my_vector;

    // Function call with pass by value
    add_elements_passby_value(my_vector, 10); 
    for (auto value : my_vector){
        std::cout<< value << ' '; //prints nothing
    }
    std::cout<<'\n';

    // Function call with pass by reference
    add_elements_using_reference(my_vector, 10); 
    for (auto value : my_vector){
        std::cout<< value << ' ';
    }

    std::cout<<'\n';
    return 0;
}