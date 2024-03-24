#pragma once
#include <iostream>
////////////////////////////////////////////////// Base class ////////////////////
// Base class for a product from the factory
class Product {

public:
    virtual void print_product() = 0;
};

/////////////////////////////////////////////////// Concrete sub class ////////////////
// Implementation 1 of the abstract class
class Product1 : public Product {

public:
    void print_product() {
        std::cout << "Hello from product 1." << std::endl;
    }
};


// Implementation 2 of the abstract class
class Product2 : public Product {

public:
    void print_product() {
        std::cout << "Hello from product 2." << std::endl;
    }
};
