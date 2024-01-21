def smart_divide(func):
    def inner(a, b):
        print("I am going to divide", a , " and" , b)
        if b == 0:
            print("Whoops! Divide by zero error!")
            return
        return func(a, b)
    return inner

@smart_divide
def divide(a, b):
    print(a/b)

print(divide(4,0)) # smart_divide decorator vitra vayeko inner() is called.
print(divide(5,5))
