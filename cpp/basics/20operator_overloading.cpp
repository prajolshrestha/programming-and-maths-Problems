#include <iostream>

struct Point {
    // Data members
    int x;
    int y;

    // member functions (methods)
    Point (int new_x, int new_y) : x(new_x), y(new_y) {}

    // operator overloading
    // Task 1:
    Point operator+(const Point &rhs) {
        return Point( x + rhs.x, y + rhs.y);
    }

    // Task 2:
    // modifying object p1
    Point& operator+=(const Point &rhs) {
        x += rhs.x;
        y += rhs.y;
        return *this; // 'this' is pointer to current object. so, we dont want to return pointer but some point by reference & we derefrence pointer
    }

    void print() {
        std::cout<<"x= "<<x<<'\n';
        std::cout<<"y= "<<y<<'\n';

    }
};

int main() {
    // Object instantiation
    Point p1(10,20);
    Point p2(30, 40);

    // Task 1:
    // Lets add two points but compiler doesn't know what '+' operator is
    Point p3 = p1 + p2; // Point p3 = p1.operator+(p2)
    p3.print();

    // Task 2:
    p1 += p2;
    p1.print();
    

    return 0;
}