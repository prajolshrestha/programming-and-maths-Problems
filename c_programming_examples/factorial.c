#include <stdio.h>

// Function declaration
int factorial(int n);

int main(){
    //variable
    int num, fact;

    //user input
    printf("Enter a number to find its factorial: ");
    scanf("%d", &num);

    //function call
    fact = factorial(num);

    //Output
    printf("The factorial of %d is %d.\n", num,fact);

    return 0;
}

// Recursive function to calculate factorial
int factorial(int n){
    if(n == 0){
        return 1;
    }else{
        return n * factorial(n-1);
    }
}