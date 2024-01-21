fn main() {
    // Rust is statically typed
    // Generally, the types are inferred
    let x = 32;
    println!("The value of x is {}", x);

    // By default, variables are immutable!
    //x = 16; // This is ILLEGAL!

    // We must directly make a variable mutable using "mut"
    let mut y = 16;
    println!("The old value of y is {}", y);
    y = 10;
    println!("The new value of y is {}", y);

    // Constants are immutable by default
    // must have its type specified
    // Underscores can be inserted into numeric literals for easy reading!
    const MAX_VALUE: u32 = 100_000;
    println!("Constant value MAX_VALUE= {}", MAX_VALUE);
}
