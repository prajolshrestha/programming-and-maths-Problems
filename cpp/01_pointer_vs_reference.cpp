// Object vs Pointer vs Reference


/*
POINTERS:
Calling objects by value can be slow and may consume a lot of resources. It is a rather inflexible approach. 
We can achieve better results by introducing a small indirection and forward pointers to objects, 
instead of the objects themselves to functions.

This function receives a pointer to a memory location that holds a variable of type integer. 
Everytime we want to access the actual value stored at that memory location, 
we have to dereference the pointer (*number) first.

Problem:
One large problem when working with pointers is the fact that we do not get any guarantee that 
the pointer actually points to a valid location in memory. The common way for functions to indicate 
an error state when they normally would return a pointer is to return the so-called null pointer 
(i.e. NULL in C or nullptr in modern C++). The C Standard specifies that this special value is actually the value 0, hence its name.
*/

void increment(int *number) {
    *number = *number + 1;
}

/*
Solution: REFERENCE
In order to prevent these problems of pointers, C++ allows us to address objects in a different, more secure way, namely by references. 
A reference is an alias for a certain object, meaning that a reference always points to a valid object.

As you can see, a reference is denoted by the & character. Unlike pointers, we do not have to derefence references, 
as they are an alias of their respective object, 
that is, the name number behaves just as if it was the actual object that the reference points to. 

It produces the exact same output as in our previous pointer-based example.
*/
void increment_ref(int &number) {
    number = number + 1;
}