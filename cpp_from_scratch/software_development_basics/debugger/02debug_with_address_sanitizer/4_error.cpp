// An example of reading uninitialized values (off-by-one jastai)

#include <algorithm>
#include <iostream>
#include <random>

int main() {
    // create some arrays
    int N = 1 << 10;
    float *a = new float[N];
    float *b = new float[N];

    // create a random number generator
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution dist(0.0f, 1.0f);

    // Generate some random numbers
    std::generate(a, a + N - 1, [&] {return dist(mt);}); // bug xa : uninitialized value
    std::generate(b, b + N - 1, [&] {return dist(mt);}); // bug xa

    // Do vector addition
    for (int i = 0; i < N; i++){
        a[i] = a[i] + b[i];
    }

    // PRint the final result
    std::cout << a[N - 1] << '\n'; // uninitialized value was created by heap allocation
    
    // Free the memory
    delete[] a;
    delete[] b;

    return 0;

}