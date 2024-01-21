"""
    an iterable is an object that can be iterated over using a for loop or other iteration constructs. 
    
    To be considered an iterable, 
    an object must have an __iter__ method that returns an iterator object. 
    The iterator object itself should have a __next__ method 
    to define how to retrieve the next element from the iterable
"""
class MyRange:
    
    '''
        MyRange class is designed to be an iterable that represents a range of numbers. 
       
        __init__ method to initialize the range.
        __iter__ method to return 'self'. (ie, returns object itself) Hence, MyRange object itself is the iterator.
        __next__ method to define how to retrive the next number.

    '''
    def __init__(self, start, end, step=1):
        self.start = start
        self.end = end
        self.step = step
        self.current = start
    
    def __iter__(self):
        '''
            __iter__ method allows the MyRange object to be treated as an iterable.
             
             When you create an instance of MyRange, 
             such as my_range = MyRange(1, 10, 2), 
             you can use it in a for loop because 
             it has an __iter__ method that returns itself.
        '''
        return self  # Return the iterator object (usually 'self')

    def __next__(self):
        # Using Chained conditionals
        if self.step > 0 and self.current >= self.end:
            raise StopIteration  # Stop iteration when current is greater than or equal to end
        elif self.step < 0 and self.current <= self.end:
            raise StopIteration  # Stop iteration when current is less than or equal to end
        else:
            result = self.current  # Store the current value
            self.current += self.step  # Update the current value for the next iteration
            return result  # Return the current value

# Using For loop
print('Using For Loop')
my_range = MyRange(1, 10, 2) # iterator object 
print(my_range) # prints object
for num in my_range:
    '''
    The for loop, when iterating over my_range, 
    repeatedly calls the __next__ method to retrieve the next number in the range 
    until StopIteration is raised
    '''
    print(num)

# Using While Loop
print('Using While Loop')
my_range = MyRange(1, 10, 2)
iterator = iter(my_range)
while True:
    try:
        num = next(iterator)
        print(num)
    except StopIteration:
        break 

# Using Comprehensions(list, dict, set)
print('Using Comprehension')
my_range = MyRange(1, 10, 2)
squared_numbers = [num ** 2 for num in my_range]
print(squared_numbers)

# Converting to a list
print('Converting to list')
my_range = MyRange(1, 10, 2)
numbers_list = list(my_range)
print(numbers_list)

# Using other iteration Functions
my_range = MyRange(1, 10, 2)
doubled_numbers = list(map(lambda x: x*2, my_range)) #map applies a function to each element
filtered_numbers = list(filter(lambda x: x % 2 == 0, doubled_numbers)) # filter filters elements based on a condition 
print(f'doubled numbers: {doubled_numbers}')
print(f'filtered numbers: {filtered_numbers}')





