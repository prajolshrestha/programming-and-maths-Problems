// Dynamic Polymorphysm
#include <iostream>

// Base class
struct Animal {
    // virtual function // this makes dynamic polymorphysm
    virtual void speak() {
        std::cout<<"Default speak function!\n";
    }
};

// Derived classes
struct Dog : Animal {
    // lets overload speak method (function overload)
    void speak() override{ // override is optional (prevents error)
        std::cout<<"Woof!\n";
    }
};

struct Cat : Animal {
    // lets overload speak method
    void speak() {
        std::cout<<"Meow!\n";
    }
};

int main() {
    Dog d; //dog type
    d.speak();
    Cat c; // cat type
    c.speak();

    // Static Polymorphism
    Animal &a1 = d; //animal type but at runtime behavior of derived class is preserved. (ie, its dog type)
    a1.speak(); // as we have introduced virtual function above, this function call will invoke behavior of the derived class

    Animal &a2 = c; // animal type
    a2.speak();

    return 0;
}