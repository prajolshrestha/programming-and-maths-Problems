#pragma once
#include "product.hpp"

class ProductFactory {
private:
    // Make the instance static
    static ProductFactory *instance;

    // Private constructor
    ProductFactory(){};

public:
    // Static access method to get our single instance
    static ProductFactory *get_instance();

    // Product instantation occurs through a factory method
    // Allows user code to not worry about instantiation
    Product *createProduct(int ID);
};