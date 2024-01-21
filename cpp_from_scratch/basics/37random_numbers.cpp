#include <iostream>
#include <random>

int main() {
    // Task 1:
    std::random_device rd;
    for (int i= 0; i<10; i++){
        std::cout<<rd() << ' ';
    }
    std::cout<<'\n';

    // Task 2: (std::mersenne_twister_engine) generates same random number every time (like it takes seed as input)
    //std::mt19937 mt;
    //std::mt19937 mt(42); // good for reproducability
    std::mt19937 mt(rd()); // now its again random and irreproducable

    for (int i= 0; i<10; i++){
        std::cout<<mt() << ' ';
    }
    std::cout<<'\n';

    // Task 3: (mix of rd, mt and uniform) rolling a dice
    std::uniform_int_distribution uniform(1, 6);
    for (int i= 0; i<10; i++){
        std::cout<<uniform(mt) << ' ';
    }
    std::cout<<'\n';

    return 0;
}