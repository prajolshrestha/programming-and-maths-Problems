class MyMeta(type):
    
    # Solution
    def __call__(cls, *args, **kwargs):
        print('{cls.__name__} is called with args = {args}, kwargs = {kwargs}')
     
        print('metaclass calls __new___')
        obj = cls.__new__(cls, *args, **kwargs)
        
        if isinstance(obj, cls):
            print('metaclass calls __init__')
            cls.__init__(obj, *args, **kwargs)
        
        return obj

class Parent(metaclass=MyMeta):
    def __new__(cls, name, age):
        print('new is called')
        return super().__new__(cls)
    
    def __init__(self, name, age):
        print('init is called')
        self.name = name
        self.age = age
    
parent = Parent('Kamran', 32) 

print(type(parent)) #<class '__main__.Parent'>