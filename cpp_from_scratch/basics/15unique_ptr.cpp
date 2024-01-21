#include <memory>
#include <iostream>

int main() {
    // Method 1:
    std::unique_ptr<int[]> ptr(new int[10]);

    for (int i = 0; i < 10; i++) {
        ptr[i] = i * i;
    }

    std::cout<< ptr[4] << '\n';
    std::cout<< ptr[7] << '\n';

    // Method 2: (w/o use of 'new')
    auto ptr1 = std::make_unique<int[]>(10);

    for (int i = 0; i < 10; i++) {
        ptr1[i] = i * i;
    }

    std::cout<< ptr1[4] << '\n';
    std::cout<< ptr1[7] << '\n';

    return 0;
}