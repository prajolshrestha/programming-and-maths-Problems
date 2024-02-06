global _start
_start:
    mov eax, 1
    mov ebx, 42
    sub ebx, 29
    int 0x80

; Q. How to run?
;$ nasm -f elf32 01_intro.asm -o intro.o
;$ ld -m elf_i386 intro.o -o intro
;$ ./intro
;$ echo $?