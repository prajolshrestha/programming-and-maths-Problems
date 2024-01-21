fn main() {
    // We can have infinite loops that require a break to exit from
    let mut id = 0;
    loop {
        // Here we use a condition for oir break
        if id == 10 {
            break;
        }

        // Otherwise, keep incrementing
        id += 1;
    }
    println!("The value of id is {}", id);

    // We can do the same with while loop
    // Here we count back down to 0
    while id != 0 {
        id -= 1;
    }
    println!("The value of id is {}", id);

    // What about traversing an array?
    // We can use a for loop ( similar to python or ranged-based ones in c++)
    let array = [1, 2, 3,4, 5];
    for element in array.iter() {
        println!("Element from array: {}", element);
    }

    // We can also use ranges in our loops ( similar to python)
    for number in 1..6{
        println!("Number from range: {}", number)
    }
}

