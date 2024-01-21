// Basics of function calls in Rust


// Function with no params, no returns! ==> Rust doesn't care they are defined or not!
fn print_func() {
    println!("This is a print from a function!");
}

// Function that takes two integers and prints them out!
fn arg_func(x: i32, y: i32) {
    println!("Value of x is {}, and value of y is {}", x, y);
}

// Function that takes an int and,  returns an ent
fn return_func(x: i32) -> i32 {
    x + 1 // Return statements don't end in  semicolons! beacause statement dont return a value!
}

fn main() {
   print_func();
   arg_func(2,3);
   let x = return_func(2);
   println!("The returned value of x is {}", x);
}
