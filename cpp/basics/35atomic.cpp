#include <iostream>
#include <thread>
#include <atomic>

int main() {
    std::atomic<int> counter = 0;
    auto work = [&counter]() {
        for (int i=0; i<100000; i++){ // problem when iteration is large: data race // solution: std::atomic
            counter += 1;
        }
    };

    std::thread t1(work);
    std::thread t2(work);
    t1.join();
    t2.join();

    std::cout<<counter<<'\n';
    return 0;
}