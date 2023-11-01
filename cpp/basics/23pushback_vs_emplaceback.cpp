// Push_back vs emplace_back

#include <iostream>
#include <vector>
#include <utility>

struct S {

    int x;

    // Constructor
    S(int new_x) :x(new_x) {
        std::cout<<"Constructor!\n";
    }

    // Copy Construtor, so there will be no "move constructor"!
    S(const S &s) : x(s.x) {
        std::cout<< "Copy constructor!\n";
    }

    // Move constructor(takes rvalue)
    S(S &&s) : x(s.x) {
        std::cout<<"Move Constructor!\n";
    }
};

int main() {
    std::vector<S> my_vector;
    //S s(10);

    // push_back
    //my_vector.push_back(s); // we gave lvalue
    //my_vector.push_back(std::move(s)); // we gave lvalue but with the use of move it acts as rvalue
    //my_vector.push_back(S(10)); // we gave rvalue

    // emplace_back (for performance)
    my_vector.emplace_back(10); // no copy & move. We are just constructing our object of type S directly inside of our vector.
   
    return 0;
}