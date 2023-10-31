#include <iostream>

class Point {
    //Default access specifier : Private
    // Data members
    int x;
    int y;

    // access specifier
    public: 
        // setters : interfaces for setting or interacting with data members
        void set_x_y(int new_x, int new_y) {
            // it acts as interface to check the data is valid or not.
            x = new_x;
            y = new_y;
        }

        // member functions (methods)
        void print() {
            std::cout<<"x= "<<x<<'\n';
            std::cout<<"y= "<<y<<'\n';
        }
};

int main() {
    // Object instantiation
    Point p1;
    p1.set_x_y(10, 20);
    p1.print();


    return 0;
}