#include <iostream>

struct Point {
    // Data members
    int x;
    int y;

    // member functions (methods)
    // Constructor
    Point(int new_x, int new_y) : x(new_x), y(new_y) {}
    
    // Copy constructor // also defined automatically by compiler!
    Point(const Point &p) { // const for safety (read only but cant modify existing pointer)
        std::cout<<"Running our copy constructor!\n";
        x = p.x;
        y = p.y;
    }

    // sometimes we dont want "copy constructor" so it deletes default copy constructor 
    Point(const Point &p) = delete;

    void print() {
        std::cout<<"x= "<<x<<'\n';
        std::cout<<"y= "<<y<<'\n';

    }
};

int main() {
    // Object instantiation
    Point p1(10,20);
    p1.print();

    // create new object based on existing object p1.
    Point p2 = p1; 
    p2.print();

    return 0;
}