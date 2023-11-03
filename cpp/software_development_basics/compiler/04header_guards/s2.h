// This is struct

#include "s1.h" // this generates redefination error of 'struct s1'

struct S2 {
    //S2 body
    S1 my_s1; // S1 as data member 
};