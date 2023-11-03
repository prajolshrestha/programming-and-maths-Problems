// This is struct

// Method 1:
// Wrap the defination of struct within header guard

//#ifndef S1_H // if something is not defined
//#define S1_H
//struct S1 {
    //S1 body
//};
//#endif

// method 2:
#pragma once
struct S1 {
    //S1 body
};