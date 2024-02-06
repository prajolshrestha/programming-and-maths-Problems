#include <stdio.h>
#include "add42.h"

int main() {
    int result;
    result = add42(30);         // 30 + 42 = 72
    printf("Result: %i\n", result);
    return 0;
}

// Assembly + C
// Assemble (32 bit)
// $ nasm -f elf32 add42.asm -o add42.o
// Link
// $ gcc -m32 add42.o main.c -o add42

// Only C
// $ gcc add42.c main.c -o add