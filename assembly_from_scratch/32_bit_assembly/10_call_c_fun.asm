global main     ;C defines _start and expects our code to provide a label main function
extern printf   ;external symbol 

section .data 
    msg db "Testing %i...", 0x0a, 0x00      ;0x00 is null terminator

section .text
main:
    push ebp
    mov ebp, esp

    push 123    ; reverse order ma xa
    push msg
    call printf
    
    mov eax, 0
    mov esp, ebp
    pop ebp
    ret