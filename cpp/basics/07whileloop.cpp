#include <iostream>

int main() {
    int work_items = 10;
    while (work_items > 0){
        work_items -= 1;

        std::cout<< work_items<<'\n';
    }

    return 0;
}