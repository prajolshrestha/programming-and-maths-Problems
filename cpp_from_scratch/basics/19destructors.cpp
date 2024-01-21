#include <iostream>
#include <memory>

struct IntArray {
    // Data members // raw pointer
    int *array; // to store dynamically allocated array of integers

    // member functions (methods)
    //  constructor
    IntArray(int size) {
        array = new int[size]; // dynamically allocate memory
    }

    // destructor : to free memory (as there is memory leak going on)
    ~IntArray() {
        std::cout<<"Running our destructor"<<'\n';
        delete[] array;
    }

   
};

struct IntArray2 {
    // Data members 
    std::unique_ptr<int[]> array;

    // member functions (methods)
    //  constructor
    IntArray2(int size) : array(new int[size]) {}

    // we dont need to worry about writing destructor. (unique_ptr handles it)

};


int main() {
    
    // Using raw pointer
    IntArray a(10);
    a.array[0] = 10;
    std::cout<<a.array[0]<<'\n';

    // Using unique_ptr
    IntArray2 b(10);
    b.array[0] = 20;
    std::cout<<b.array[0]<<'\n';

    return 0;
}