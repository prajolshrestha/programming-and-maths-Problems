class MyClass:

    def __init__(self, value) -> None:
        '''
            Constructor Method that initializes an object when it is created.
        '''
        self.value = value
        

    def __str__(self) -> str:
        '''
            Defines the string representation of an object 
            when using str(obj) or print(obj)
        '''
        return f"MyClass with value:{self.value}"
    
    def __repr__(self) -> str:
        '''
            efines the "official" string representation of an object 
            used by repr(obj) and in interactive environments.
        '''
        return f"Myclass({self.value})"

obj = MyClass(42)
print(obj) # str
print(repr(obj)) # repr


class MyList:

    def __init__(self,items) -> None:
        self.items = items

    def __len__(self):
        '''
            Used to define the length of an object when using len(obj).
        '''
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]
    
    def __setitem__(self, index, value):
        self.items[index] = value
    

obj2 = MyList([1,2,3,4])
print(len(obj2)) # len  

print(obj2[2]) # getitem
obj2[2] = 42 #setitem
print(obj2[2]) #getitem
    


