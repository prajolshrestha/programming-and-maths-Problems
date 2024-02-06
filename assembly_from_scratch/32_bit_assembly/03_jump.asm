global _start

section .text
_start:
    mov ecx, 101
    mov ebx, 42   ; exit status is 42
    mov eax, 1    ; sys_exit sys call
    ;jmp skip      ; jump to "skip" label ; UNCONDITIONAL JUMP
    cmp ecx, 100    ; compare ecx to 100
    jl skip         ; jump if less than
    mov ebx, 13   ; exit status is 13
skip:
    int 0x80      ; interupt to exit the program
   
