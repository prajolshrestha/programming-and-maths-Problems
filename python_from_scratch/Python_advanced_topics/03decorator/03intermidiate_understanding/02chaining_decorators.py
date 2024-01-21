"""
    To chain decorators, we can apply multiple decorators to a single function by placing them one after the other, 
    with the most inner decorator being applied first.
"""

def star(func):
    def inner(*args, **kwargs):
        print("*"*15)
        func(*args, **kwargs) # calls percent()
        print("*"*15)
    return inner

def percent(func):
    def inner(*args, **kwargs):
        print('%'*15)
        func(*args,**kwargs) # calls printer()
        print('%'*15)
    return inner

@star
@percent  # when we do not use @ symbol: ==> printer = star(percent(printer))
def printer(msg):
    print(msg)

print(printer("Hello"))