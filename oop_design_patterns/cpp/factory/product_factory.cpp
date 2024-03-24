#include "product_factory.hpp"

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