// Simple Program that modifies the contents of a vector

// Compile : g++ main.cpp -o main_opt_3_native -o3 -march=native
// Run: Measure-Command {./3}

#include <vector>

int main() {
    // Create a vector of 2^28 elements (0s)
    auto num_elements = 1 << 28;
    std::vector<int> vector(num_elements);

    // Modulo each value by 20
    for (auto value : vector) {
        value %= 20;
    }
    return 0;

}
