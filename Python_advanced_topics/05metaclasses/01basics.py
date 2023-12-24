class MyMeta(type):
    def __call__(self, *args, **kwargs):
        print('{self.__name__} is called with args = {args}, kwargs = {kwargs}')
    
class Parent(metaclass=MyMeta):
    def __new__(cls, name, age):
        print('new is called')
        return super().__new__(cls)
    
    def __init__(self, name, age):
        print('init is called')
        self.name = name
        self.age = age
    
parent = Parent('Kamran', 32) # just prints info, Problem: __new__ and __init__ is not being called!

print(type(parent)) #<class 'NoneType'>