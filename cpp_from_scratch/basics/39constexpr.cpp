#include <iostream>
#include <random>

// compute at compile time
constexpr int factorial(int n) {
    if (n<=1) {
        return 1;
    } else {
        return n * factorial(n-1);
    }
}

int main() {
    constexpr int result = factorial(5);
    std::cout<<result <<'\n';

    std::random_device rd; // non constexpr function
    int result2 = factorial(rd() % 6); // we cant use constexpr because we were relying at runtime random number generation fun rd()
    std::cout<<result2 <<'\n';
    return 0;
}