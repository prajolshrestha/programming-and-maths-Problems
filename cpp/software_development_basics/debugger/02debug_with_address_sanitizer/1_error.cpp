// An example with a memory leak

#include <algorithm>
#include <random>

int main() {
    // Create some arrays
    int N = 1<<10; // 2^10 = 1024
    float *a = new float[N]; // dynamically allocates array of 1024 floats and assigns its starting address to pointer 'a'
    float *b = new float[N];

    // Create random number generator
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution dist(0.0f, 1.0f);

    // Generate some random numbers
    // a, a+N is range
    // [&] {return dist(mt); } is lambda function, used to generate random floating number using previously defined distribution.
    // & in [&] captures variables used in the lambda function by reference, allowing access to dist and mt. 
    std::generate(a, a + N, [&] {return dist(mt); }); // to fill the array 'a' with random numbers.
    std::generate(b, b + N, [&] {return dist(mt); });

    // Do vector addition
    for (int i = 0; i < N; i++) {
        a[i] = a[i] + b[i];
    }
}

//Problem: We never free the memory!

// valgrind ./1_error
//
// HEAP Summary:
//
// LEAK Summary: 
//               2 blocks lost

// To see details of leaked memory
// valgrind --leak-check=full ./1_error
//