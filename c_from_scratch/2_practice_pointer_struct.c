#include <stdio.h> // for functions like printf and scanf
#include <stdlib.h>// for functions like malloc and free

// Define a struct called "Person" with name and age fields 
// Declares the struct directly.
struct Person{
    char name[50];
    int age;
};

// Define a function that takes a pointer to a Person struct as an argument
void change_person(struct Person *p, char *new_name, int new_age){
    // Assign new values to the name and age fields using the pointer
    sprintf(p->name,"%s", new_name);
    p->age = new_age;//it assigns the value of new_age to the age field of the Person struct using the pointer p.
}

int main(){
    // Declare a variable of type Person
    struct Person my_person;

    // Declare a pointer to a person struct and assign it to the address of my_person
    struct Person *my_pointer = &my_person;

    // Modify the fields of my_person using the pointer
    change_person(my_pointer, "John", 25);

    // Print
    printf("Name: %s\n", my_person.name);
    printf("Age: %d\n",my_person.age);

    return 0;

}