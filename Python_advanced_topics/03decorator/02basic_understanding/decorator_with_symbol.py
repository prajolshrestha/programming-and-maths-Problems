def outer(func):
    def inner():
        print('I am decorated!')
        func()
    return inner


# Lets decorate original_function() with outer() decorator using '@ outer' syntax
@outer # @ symbol with decorator ---> @ symbol helps to assign the function call to a variable  [equivalent to: decorated_func = outer(original_function)]
def original_function():
    print("I am original!")

print(original_function())