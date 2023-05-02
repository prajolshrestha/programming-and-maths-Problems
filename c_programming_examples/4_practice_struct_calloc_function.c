#include <stdio.h>
#include <stdlib.h>

// Define a struct
struct human {
    char name[50];
    int age;
};

// Function to create a new human struct
struct human* create_human() {
    // Allocate memory for human struct
    struct human* ptr = (struct human*) calloc(1, sizeof(struct human));
    //check if memory allocation failed
    if(ptr == NULL) {
        printf("Memory allocation failed!\n");
        return NULL;
    }

    // Prompt user for 'name' and 'age' using scanf
    printf("Enter name: ");
    scanf("%s", ptr->name);
    printf("Enter Age: ");
    scanf("%d", &ptr->age);

    // return pointer to new human struct
    return ptr; 
}

//Function to greet
void greet(struct human* ptr){
    printf("Hello, my name is %s and I am %d years old.\n", ptr->name, ptr->age);
}

void delete_human(struct human* ptr){
    free(ptr);
}

int main() {
    //create two human struct
    struct human* h1 = create_human();
    struct human* h2 = create_human();

    // print greeting message for each human using greet() function
    greet(h1);
    greet(h2);

    // Free memory
    delete_human(h1);
    delete_human(h2);

    return 0;

}