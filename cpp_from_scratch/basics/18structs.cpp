#include <iostream>

struct Point {
    // Data members
    int x;
    int y;

    // member functions (methods)
    void print() {
        std::cout<<"x= "<<x<<'\n';
        std::cout<<"y= "<<y<<'\n';

    }
};

int main() {
    // Object instantiation
    Point p1;
    p1.x = 10; // data member allocation using object and member access operator
    p1.y = 20;
    p1.print();

    // New object
    Point p2;
    p2.x = 80; // data member allocation using object and member access operator
    p2.y = 70;
    p2.print();

    return 0;
}