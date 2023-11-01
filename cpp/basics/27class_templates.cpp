#include <iostream>
#include <memory>

// class template: lets create different version of dynamicArray for different types T that we can specify later
template <typename T> // template with template parameter T
struct DynamicArray {
    // data members
    int size;
    std::unique_ptr<T[]> ptr; // unique_ptr managing an array of type T

    // constructor
    DynamicArray(int new_size) : size(new_size), ptr(new T[new_size]) {}

    // member functions
    void fill(T value) {
        for (int i = 0; i<size; i++){
            ptr[i] = value; // index our array through pointer'ptr' and add some value
        }
    }

    void print() {
        for(int i=0; i<size; i++) {
            std::cout<<ptr[i]<<' ';
        }
        std::cout<<'\n';
    }

};

int main() {
    // Dynamic array with type int
    DynamicArray<int> int_array(10);
    int_array.fill(5);
    int_array.print();
    
    // Dynamic array with type double
    DynamicArray<double> double_array(10);
    double_array.fill(1.23);
    double_array.print();
    
    return 0;

}