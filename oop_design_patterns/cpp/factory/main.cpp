#include <iostream>
#include "product.hpp"
#include "product_factory.hpp"

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