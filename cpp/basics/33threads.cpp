#include <iostream>
#include <thread>
#include <vector>

void print_thread_id(int id) {
    std::cout<<"Printing from the thread: "<<id<<'\n';
}

int main() {
    // Task 1:
    // Creating a thread // g++ 33threads.cpp -o threads -lpthread
    std::thread t1(print_thread_id, 0);
    t1.join(); // join with main thread

    // Task 2:
    // lets write in cleaner way (thread + join)
    std::jthread t2(print_thread_id, 0);

    // Task 3:
    // spwanning groups of threads
    std::vector<std::jthread> my_threads;
    for (int i= 0; i<3; i++) {
        my_threads.emplace_back(print_thread_id, i); // forward the args to our constructor for our jthread

    }

    return 0;
}