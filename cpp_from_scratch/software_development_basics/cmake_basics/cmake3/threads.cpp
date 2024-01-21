#include <iostream>
#include <thread>
#include <atomic>

// Two threads doing increments (protected by lock guard)
int main() {
    // Number of increments per-thread
    const int num_increments = 1 << 20;

    // Work lambda with protected increments
    int val = 0;
    std::mutex m;
    auto work = [&] {
        for (int i=0; i<num_increments; i++) {
            std::lock_guard<std::mutex> lg(m);
            val += 1;
        }
    };

    //Spawn our threads
    std::thread t0(work);
    std::thread t1(work);

    // Wait for threads to finish
    t0.join();
    t1.join();

    // Print the final result
    std::cout<<"Final value:" <<val <<'\n';
    return 0;
}