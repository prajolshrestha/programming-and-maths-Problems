#include <iostream>

struct S {
    int a;
    int b;

    // default comparision (compilar will create operator for us)
    bool operator==(const S &s) const = default;

    // 3-way comparision operator (looks like spaceship <=>)
    auto operator<=>(const S &s) const = default;

};

int main() {
    S s1 {1, 2}; //aggregat initialization
    S s2 {1, 2};

    std::cout<<(s1 == s2)<<'\n';
    std::cout<<(s1 > s2)<<'\n';

    return 0;
}