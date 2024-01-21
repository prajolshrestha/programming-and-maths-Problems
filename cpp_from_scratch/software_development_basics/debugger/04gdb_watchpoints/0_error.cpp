// A simple example of watchpoints in gdb

int main() {
    // Allocate some memory
    const int N = 1 << 10;
    auto p = new int[N];

    // set the contents in a for loop
    for (int i = 0; i<N; i++) {
        p[i] = 10;
    }

    // set the contents in a for loop
    for (int i = 0; i < N; i++){
        p[i] = 20;
    }

    // Free the memory
    delete[] p;
    return 0;
}

// lets see how the contents of the array changes over time
