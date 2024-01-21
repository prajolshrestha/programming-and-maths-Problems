#include <iostream>
using namespace std;

//Abstract class
class Shape{
    public:
        virtual void draw() = 0; //Pure virtual function (making it abstract):  it signifies that this function is intended to be overridden by derived classes 
        virtual ~Shape() {} // Virtual destructor
};

// Derrived class Rectangle inheriting from Shape
class Rectangle : public Shape{
    private: // Encapsulation (private, public, protected)
        int width;
        int height;

    public:
        Rectangle(int w, int h): width(w), height(h){} //Constructor

        void draw() override{ //polymorphism: own implementation of draw() for rectangle class
            cout<<"Drawing a Rectangle with width" << width <<" and height" << height << endl;
        }
};

int main(){
    // Creating objects using pointers and refrences
    Rectangle rect1(5,3); // rect1 is instance of rectangle class
    Shape* shapePtr = &rect1;// Pointer to address of object 'rect1'
    Shape& shapeRef = rect1;// Refrence to object 'rect1'

    // Using the objects
    rect1.draw(); // direct call functions
    shapePtr->draw();// indirectly call funcions
    shapeRef.draw();// Indirectly call functions

    return 0;
}
/*
Pointers are primarily used for:

        Dynamic memory allocation and deallocation using new and delete operators.
        Storing memory addresses, which can be used for various purposes, including data structures like linked lists and trees.
        Achieving polymorphism through pointers to base class objects and virtual functions.
        Accessing and modifying data indirectly.

References are primarily used for:

        Providing an alternative name (alias) for an existing object.
        Function parameters to avoid copying large objects (passing by reference).
        Achieving polymorphism through references to base class objects and virtual functions (similar to pointers).
        Writing more readable and cleaner code when you don't need to work with pointers directly.
*/

/*
Polymorphism:
It's a concept that allows objects of different classes to be treated as objects of a common superclass. 
Polymorphism enables you to write code that can work with objects of various classes in a uniform way.
*/