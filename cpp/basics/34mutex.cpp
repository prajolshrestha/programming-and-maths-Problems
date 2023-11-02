#include <iostream>
#include <thread>
#include <mutex>
#include <vector>


int main() {
    // create mutex // helps to seralize the access to std::cout in this example
    // Mutual Exclusion ie, no two thread can pass mutex lock at same time.
    std::mutex m;

    // function rewritten as lambda expression
    auto print_thread_id = [&m](int id) {

        // Method 1: (if we forget to unlock ==> we are in deadlock)
        //m.lock(); // each threads locks the mutex before proceding to print
        //std::cout<<"Printing from the thread: "<<id<<'\n';
        //m.unlock(); // after printing , threads unlocks the mutex    

        // Method 2: (we dont have to lock and unlock, it handles evth)
        std::lock_guard<std::mutex> lg(m); // constructor ma lock happens automatically, and destructor ma unlock happens automatically
        std::cout<<"Printing from the thread: "<<id<<'\n';

    };
    
    // spwanning groups of threads // Problem: multiple threads access std::cout exactly at the same time //solution: std::mutex
    std::vector<std::jthread> my_threads;
    for (int i= 0; i<3; i++) {
        my_threads.emplace_back(print_thread_id, i); // forward the args to our constructor for our jthread

    }

    return 0;
}