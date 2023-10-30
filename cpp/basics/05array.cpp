#include <array>
#include <iostream>

int main() {
     std::array<int, 3> my_array = {1,2,3};

     // std::cout<< my_array << '\n'; // This is wrong!

     // Element access
    
     std::cout<< my_array.at(0) << '\n'; // access specified element with bounds checking using at()
    
     std::cout<< my_array[1] << '\n'; // access specified element using []

     std::cout<< my_array.front() << '\n'; // access first element
    
     std::cout<< my_array.back() << '\n'; // access last element

    // Capacity

    std::cout<<"size= "<< my_array.size() <<'\n'; // returns the number of elements


     // Operations
     my_array[0] = 10; // replace first element with 10

     std::cout<< my_array.at(0) << '\n'; 

     my_array.fill(54); // fill the container with specified value

     std::cout<< my_array.front() << '\n'; 
     std::cout<< my_array.back() << '\n'; 


    return 0;
}