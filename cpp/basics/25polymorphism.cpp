#include <iostream>

// Base class
struct Animal {
    void speak() {
        std::cout<<"Default speak function!\n";
    }
};

// Derived classes
struct Dog : Animal {
    // lets overload speak method
    void speak() {
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
    // Lets interpret our Dog as our base class (as animal above). How? ==> create simple reference to base class
    Animal &a1 = d; //animal type // we created reference of type animal and we pass dog object. // as animal is being inheritated by dog // Note: we are not creating new object of type animal. but we are just creating alias to dog object.
    a1.speak(); // as we intrepreted dog object as animal

    Animal &a2 = c; // animal type
    a2.speak();

    return 0;
}