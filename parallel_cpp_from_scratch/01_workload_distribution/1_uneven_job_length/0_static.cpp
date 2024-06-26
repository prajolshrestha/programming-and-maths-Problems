#include <tbb/parallel_for.h>

#include <vector>
#include <algorithm>
#include <random>

int main () {
    
    // Create a random number generator
    std::random_device rd;
    std::mt19937 mt(rd());

    // calculate number of elements per bin
    int num_work_items = 1 << 18;
    int n_bins = 4;
    int elements_per_bin = num_work_items / n_bins;

    // create 4 distribution
    std::uniform_int_distribution bin_1(1, 25);
    std::uniform_int_distribution bin_2(25, 50);
    std::uniform_int_distribution bin_3(50, 75);
    std::uniform_int_distribution bin_4(75, 100);


    // Generate random numbers
    std::vector<int> work_items;
    std::generate_n(std::back_inserter(work_items), elements_per_bin,[&] { return bin_1(mt); });
    std::generate_n(std::back_inserter(work_items), elements_per_bin,[&] { return bin_2(mt); });
    std::generate_n(std::back_inserter(work_items), elements_per_bin,[&] { return bin_3(mt); });
    std::generate_n(std::back_inserter(work_items), elements_per_bin,[&] { return bin_4(mt); });

    // Process all elements in a parallel_for loop
    tbb::parallel_for(
        tbb::blocked_range<int>(0, num_work_items),
        [&] (tbb::blocked_range<int> r) {
            for (int i=r.begin(); i < r.end(); i++){
                std::this_thread::sleep_for(std::chrono::microseconds(work_items[i]));
            }
        },
        tbb::static_partitioner());

    return 0;
}