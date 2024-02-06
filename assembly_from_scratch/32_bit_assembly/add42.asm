global add42

add42:
    ;Prologue
    push ebp
    mov ebp, esp 

    mov eax, [ebp+8]
    add eax, 42

    ;Epilogue
    mov esp, ebp
    pop ebp 
    ret
