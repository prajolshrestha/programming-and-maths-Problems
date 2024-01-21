#include <numeric>
#include <vector>
#include <execution> //execution policy included (sequenced_policy, parallel_policy, parallel_unsequenced_policy, unsequenced_policy)

int main() {
    std::vector<int> my_vector(1 << 30); // 2^30 ie, 4 GB total integers
    //auto r = std::reduce(my_vector.begin(), my_vector.end(), 0); // normal code
    auto r = std::reduce(std::execution::par_unseq, my_vector.begin(), my_vector.end(), 0); // parallelism (parallel and vectorized)

    //REQUIRES ltbb library
    return r;
}