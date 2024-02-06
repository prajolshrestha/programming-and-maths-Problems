global _start

section .data                       ; inline data into the program that we can ref. in the code by name
    msg db "Hello, world!", 0x0a    ; created a string of bytes named msg, which has helloworld followed by 0x0a, ie 10, code for new line 
    len equ $ - msg                 ; determining length of the string (by sub loc of start of the string from the location after the string)
                                    ; we could inline the length as an integer which is 14 in this case
                                    ;But, doing like our code , allows us to change the string without having to count and update the length manually

section .text ; This section is where our code lives.
_start:
    mov eax, 4   ;sys_write system call
    mov ebx, 1   ;stdout file descriptor
    mov ecx, msg ;bytes to sys_write  ; it holds string pointer
    mov edx, len ;number of bytes to write ; it holds string length
    int 0x80     ; perform system call ; interrupt
    
    mov eax, 1   ;sys_exit system call
    mov ebx, 0   ;exit status is 0 ; means program run successfully
    int 0x80

; vim /usr/include/asm/unistd_32.h  --> this header file provides access to various POSIX OS APIs
; man 2 write      --> to see info about sys call