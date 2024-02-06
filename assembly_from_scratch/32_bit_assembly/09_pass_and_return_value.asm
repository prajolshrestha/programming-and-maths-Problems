; 21+21 = 42 using a function
global _start

_start:
    push 21     ;push 21 to stack 
    call times2 ;push return address to stack
    mov ebx, eax
    mov eax, 1
    int 0x80


times2:
    push ebp            
    mov ebp, esp

    mov eax, [ebp + 8]  
    add eax, eax 

    mov esp, ebp    ;esp=ebp=20 address
    pop ebp         ;ebp = 123
    ret
