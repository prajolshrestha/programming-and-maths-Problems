// An example to run through with the GDB TUI

#include <algorithm>
#include <random>

int main() {
    // Create some arrays
    int N = 1 << 10;
    float *a = new float[N];
    float *b = new float[N];

    // Create random number generator
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution dist(0.0f, 1.0f);

    // Generate some random numbers
    std::generate(a, a + N, [&] {return dist(mt); });
    std::generate(b, b + N, [&] {return dist(mt); });

    // Do vector addition
    for (int i = 0; i > N; i++) {
        a[i] = a[i] + b[i];
    }

    // free our memory
    delete[] a;
    delete[] b; 
    
    return 0;
}

