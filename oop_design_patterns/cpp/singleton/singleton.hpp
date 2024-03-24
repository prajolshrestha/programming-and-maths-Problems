#pragma once
//#ifndef SINGLETON_HPP
//#define SINGLETON_HPP

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

//#endif 