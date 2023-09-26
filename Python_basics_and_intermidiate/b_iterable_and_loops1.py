"""
An iterable in Python is any object capable of returning its elements one at a time. 
You can iterate (loop) over the elements of an iterable using a for loop or other iteration constructs.
"""
# List
my_list = [1, 2, 3, 4, 5]
for item in my_list:
    print(item)

# Tuple
my_tuple = (10, 20, 30)
for item in my_tuple:
    print(item)

# String
my_string = "Hello, World!"
for char in my_string:
    print(char)

# Dictonary
my_dict = {'a': 1, 'b': 2, 'c': 3}
for key in my_dict:
    print(key)

my_dict = {'a': 1, 'b': 2, 'c': 3}
for key, value in my_dict.items():
    print(f"Key: {key}, Value: {value}")

# Set
my_set = {1, 2, 3, 4, 5}
for item in my_set:
    print(item)

# Generator
def my_generator():
    '''
    Generators are a way to create custom iterators. 
    They are defined using functions with yield statements.
    '''
    yield 1
    yield 2
    yield 3

for item in my_generator():
    print(item)

#Iterators: range, enumerate
for num in range(1, 6):
    print(num)

for index, value in enumerate(['a', 'b', 'c']):
    print(f"Index: {index}, Value: {value}")

# Files
with open('example.txt', 'r') as file:
    for line in file:
        print(line.strip())


