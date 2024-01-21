// This program calculates the definite integral of the function
// f(x) = exp(-x) * sin(x) from x = 0 to x = pi/2 
// using the trapezoidal rule.

/*The integral function takes the lower and upper limits of integration (a and b),
a pointer to the function to be integrated (f), and the number of trapezoids 
to use in the integration (n) as input arguments, and returns the numerical value of the integral.

The main function sets up the integration problem by defining 
the limits of integration and the function to be integrated (f), 
and then calls the integral function to calculate the value of the integral. 
Finally, the program prints out the result of the integration using printf. */

#define M_PI 3.14159265
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function declaration
double integral(double a, double b, double (*f)(double), int n); //double (*f)(double) is a function pointer.
/*This syntax indicates that f is a pointer to a function that takes a double argument
 and returns a double value. 
 In other words, f points to a function that we can call using f(x) 
 to get the value of the function at x.*/


//Function to integrate
double f(double x){
    return exp(-x) * sin(x);
}

// main function
int main(){
    // Variables
    double a = 0.0, b = M_PI/2.0, result;
    int n = 1000000;

    // Integration
    result = integral(a, b, &f, n);

    // Output
    printf("The integral of function is %f.\n", result);

    return 0;

}

// Function to numerically integrate using trapezoidal rule
double integral(double a, double b, double (*f)(double), int n){
    double h = (b-a) / (double)n;//(width of each subinterval )==>This line calculates the width of each subinterval h by subtracting the lower limit of integration a from the upper limit of integration b, and dividing the result by the number of subintervals n
    double sum = 0.5 * (f(a) + f(b));//(area of the trapezoid that connects the two endpoints of the interval)==>This line initializes the sum variable to half the sum of the function values at the lower limit of integration a and the upper limit of integration b.
    int i;
    //loop calculates the sum of the areas of the trapezoids
    // corresponding to the subintervals between the lower and upper limits of integration
    for (i =1; i < n; i++){
        double x = a + i * h;//In lower limit (index * width of each interval) is added
        sum += f(x);
    }
    return h * sum;
}