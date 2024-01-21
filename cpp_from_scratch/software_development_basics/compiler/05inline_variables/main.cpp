#include <iostream>

#include "function.h"
#include "global.h"

int main() {
    global_value += 10; // redefination error //sol: inline variable
    update_value(10);
    std::cout<< global_value << '\n';
    return 0;
}