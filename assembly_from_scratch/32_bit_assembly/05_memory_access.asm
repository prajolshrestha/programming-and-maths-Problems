global _start
section .data
    addr db "yellow"    ; addr points to beginning of some address of memory

section .text
_start:
    mov [addr], byte 'H'        ; access first address 
    mov [addr+5], byte '!'      ;using offset to access different part of address 
    mov eax, 4  ; sys_write sys call
    mov ebx, 1  ;stdout file descriptor
    mov ecx, addr ;bytes to write
    mov edx, 6    ;number of bytes to write
    int 0x80
    mov eax, 1  ;sys_exit call
    mov ebx, 0  ;exit status is 0
    int 0x80