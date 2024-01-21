// An example of watching values with gdb

#include <iostream>
#include <random>

int main() {
    
    // Create random number generator
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution dist(0, 100);

    // Run a performing some calculations
    int sum = 0;
    while (true) {
        // Get a divisor
        auto divisor = dist(mt);

        // Add some value to sum
        // This has a bug! Our random number generator can return 0! ==> division by 0 error huna sakxa
        sum += 10 % divisor;

        // Break out with some condition
        if (sum > 200) break;
    }

    // Print out sum
    std::cout << sum << '\n';
    return 0;
  
}

// Program doesn't crash all the time. This is non-deterministic error.

// gdb ./1_error
// run   // several time garda , error vayeko bela rokkine
// list
// print(sum)
// ptint(divisor)  // must be 0!


// instead of typing 'run' several time and find error, we can do this

/////// Commands for breakpoint

// break 28
// commands
//   >run
//   >end
// info breakpoints

//////// conditional breakpoint

// break 21 if divisor==0
// commands
//   >set divisor = 1
//   >print("Patching divisor!")
//   >continue
//   >end
// info breakpoints

// run