global _start

section .text
_start:
    mov ebx, 1 ; start ebx at 1
    mov ecx, 4 ; number of iterations    ie 2^4 = 16
label:          ;start of the loop
    add ebx, ebx ; 
    dec ecx     ; ecx -= 1 ; decrement
    cmp ecx, 0
    jg label
    mov eax, 1 ; sys_exit call
    int 0x80

; output = 16