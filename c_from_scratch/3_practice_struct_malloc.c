/*Both malloc and calloc are used to allocate memory dynamically during runtime
 , but they differ in how they allocate and initialize the memory.
 
 -> malloc allocates a block of memory of a specified size 
    and returns a pointer of type void to the first byte of the block.
    The content of the memory block is uninitialized, meaning it may contain garbage values.
 
 ->calloc is similar to malloc, but it also initializes the memory to zero. 
   It takes two arguments: the number of elements to allocate and the size of each element. 
   It then calculates the total size required for the allocation and returns a pointer of type void to the first byte of the block.
 */
#include <stdio.h>
#include <stdlib.h>

// use a typedef to create an alias for the struct type. (more concise and easier to read.)
typedef struct{
    int id;
    char name[50];
    int age;
}Person;

int main() {
    //use malloc to allocate memory for "a single Person struct"
    Person *p1 = (Person*) malloc(sizeof(Person));//dynamically allocates a block of memory 
                                                 //of size equal to the size of the "Person" struct 
                                                 //using the sizeof operator and returns a pointer to it

    // Use pointer to access struct number and assigning values to struct
    p1->id = 1;
    strcpy(p1->name, "John");
    p1->age = 25;

    // print all value
    printf("Person Id: %d\n",p1->id);
    printf("Person Name: %s\n",p1->name);
    printf("Person Age: %d\n",p1->age);

    // Use calloc to allocate memory for "array of Person structs"
    Person *p2 = (Person*) calloc(2,sizeof(Person));
    
    // Use pointer to access array elements and assigning values to struct
    p2[0].id = 2;
    strcpy(p2[0].name, "Mary");
    p2[0].age = 30;

    p2[1].id = 3;
    strcpy(p2[1].name, "Bob");
    p2[1].age = 31;
    
    //print all value
    printf("List of Persons: \n");
    for(int i = 0; i<2; i++) {
        printf("Person %d ID: %d\n", i+1, (p2+i)->id);
        printf("Person %d Name: %s\n", i+1, (p2+i)->name);
        printf("Person %d Age: %d\n", i+1, (p2+i)->age);

    }

    //free memory
    free(p1);
    free(p2);

    return 0;
}


