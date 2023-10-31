#include <iostream>

struct Point {
    // Data members
    int x;
    int y;

    // member functions (methods)
    
    //// Method 1:
    //Point() { // constructor
    //    x = 10;
    //    y = 20;
    //} 

    //// Method 2:
    //Point () : x(10), y(20) {}

    // Method 3:
    //Point(int new_x, int new_y){
    //    x = new_x;
    //    y = new_y;
    //}

    // Method 4:
    Point() = default; // default constructor
    Point(int new_x, int new_y) : x(new_x), y(new_y) {}

    void print() {
        std::cout<<"x= "<<x<<'\n';
        std::cout<<"y= "<<y<<'\n';

    }
};

int main() {
    // Object instantiation
    Point p1(5,7);
   
    p1.print();


    return 0;
}