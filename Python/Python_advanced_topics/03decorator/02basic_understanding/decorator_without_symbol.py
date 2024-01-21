def outer(func): # decorator
    def inner():
        print("I got decorated (ie, original func is modified!)")
        func()
    return inner # call inner function

def original_function():
    print("I am original!")


decorated_func= outer(original_function) # assign function call to a variable # Later, this can be omitted using @ symbol
print(decorated_func())