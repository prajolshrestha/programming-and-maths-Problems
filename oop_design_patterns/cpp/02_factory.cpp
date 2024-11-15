// Factory creational design pattern

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

//////////////////////// Simple factory for products //////////////////
// Commonly implemented as singleton
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

// Create a product on the caller's behalf
Product *ProductFactory::createProduct(int ID) {
    // Create a product of type 1
    if(ID == 0){
        return new Product1();
    // Create a product of type 2
    }else if (ID == 1){
        return new Product2();
    // Return nullptr if no matching type
    }else {
        return nullptr;
    }
}

// Initialize static class member to be a nullptr
ProductFactory* ProductFactory::instance = nullptr;

// Returns the single Product Factory instance. Creates it if  it doesnot already exist
ProductFactory* ProductFactory::get_instance() {
    // check if the static instance variable is null still (lazy instantiaion: don't create until needed)
    if (!instance){
        // If not, create the first singleton instance
        instance = new ProductFactory();
    }
    // and retrun it
    return instance;

}

int main() {
    // Create the first instance of the factory singleton
    ProductFactory *pf = ProductFactory::get_instance();

    // Create a pointer parent class
    Product *p;

    // Create a product of type 0
    p = pf->createProduct(0);
    p->print_product();

    // Create a product of type 1
    p = pf->createProduct(1);
    p->print_product();

    return 0;

}