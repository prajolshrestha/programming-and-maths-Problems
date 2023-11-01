#include <iostream>

struct Base {
    int x;
    int y;

    Base(int new_x, int new_y) : x(new_x), y(new_y) {}

    void print_x_y() {
        std::cout<<"x= "<<x<<'\n';
        std::cout<<"y= "<<y<<'\n';

    }
};

// lets specialize base class on our own way
struct Derived1 : Base {
    int z;

    Derived1(int new_x, int new_y, int new_z) : Base(new_x, new_y), z(new_z) {}

    void print_z() {
        std::cout<<"z= "<<z<<'\n';
    }
};

int main() {
    Derived1 d1(10,20,30);
    d1.print_x_y();
    d1.print_z();
    return 0;
}