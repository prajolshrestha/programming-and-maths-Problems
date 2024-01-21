// Abstract class
#include <iostream>

// Base class
struct Animal {
    // pure virtual function : we are not allowed to instantiate but allowed to inherit from it.
    virtual void speak() = 0;
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
    a1.speak(); 

    Animal &a2 = c; // animal type but still invokes cat type
    a2.speak();

    //Animal a; // we cannot make object of base class
    //a.speak();

    return 0;
}