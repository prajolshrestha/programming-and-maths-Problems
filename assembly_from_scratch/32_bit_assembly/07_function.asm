 global _start
 
 _start:
    call func
    mov eax, 1      ; jump garera yeta aauxa from jmp eax
    int 0x80

func:
    mov ebx, 42
    pop eax   ;pop location of code off the stack and put it into eax 
    jmp eax     	;jump to loc stored in eax (ie, next instruction)

    ;ret  ;shortcut

