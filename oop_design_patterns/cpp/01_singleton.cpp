// Topic: Singleton creational design pattern in c++
//

#include <iostream>

// Singleton class to show design pattern
class Singleton{
private:
    // Make the singleton instance static (only one ever)
    static Singleton *instance;     

    // Private constructor to ensure only a single instantiation
    Singleton(){};

public:
    // Static access method to get our single instantiation
    static Singleton *getInstance();

};

// Initialize static class member to be a nullptr
Singleton* Singleton::instance = nullptr;

// Return single singleton instance. (Create only if instance doesnot exist)
Singleton *Singleton::getInstance(){
    // If instance == nullptr, create singleton object (ie, don't create until needed)
    if(!instance){
        instance = new Singleton();
    }
    return instance;
}

int main(){
    // ILLEGAL! Constructor is private!
    // Singleton *test = new Singleton();

    // Creates a new singleton, and returns a pointer to it.
    Singleton *simple_singleton_1 = Singleton::getInstance();

    // only returns a pointer to the single singleton instance
    Singleton *simple_singleton_2 = Singleton::getInstance();

    std::cout<< "Singleton 1: "<< simple_singleton_1<< "Singleton 2: "<<simple_singleton_2 << std::endl;

    return 0;
}



