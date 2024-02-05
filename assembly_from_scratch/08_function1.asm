global _start

_start:
    call func
    mov eax, 1
    mov ebx, 0
    int 0x80

func:
    ;Prologue
    push ebp        ; push the prev. value of ebp on to satck when you enter function   ; to preserve old value
    mov ebp, esp    ; move stack pointer to base pointer ie save the top of the stack address
    sub esp, 2      ; its like allocating 2 bytes
    
    mov [esp], byte 'H'
    mov [esp+1], byte 'i'
    
    ; sys write call
    mov eax, 4
    mov ebx, 1
    mov ecx, esp    ;bytes to write
    mov edx, 2      ; number of bytes to write
    int 0x80

    ;Epiloge
    mov esp, ebp    ; restore the value we just saved above // this effectively deallocates the space
    pop ebp     ; pop before you return
    ret
