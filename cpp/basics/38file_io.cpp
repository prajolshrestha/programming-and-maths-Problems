#include <fstream>
#include <iostream>

int main() {
    // Write data to file

    //std::ofstream output("data.txt");
    //for (int i=0; i<10; i++) {
    //    output << i * i <<' '; // acts similar to  std::cout
    //}
    //output << '\n';

    // read data from file
    std::ifstream input("data.txt");
    int data;
    while(input >> data) { 
        std::cout<< data<<' ';
    }
    std::cout<< '\n';
    return 0;
}