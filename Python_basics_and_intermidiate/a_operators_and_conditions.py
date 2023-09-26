"""
    These are some of the common operators, conditions and chained conditionals in Python. 

    They are used for arithmetic calculations, comparisons, logical operations, 
    identity checks, and membership tests, among other things.

    Chained conditionals allow you to create complex decision-making structures
"""

# 1. Arithmetic Operators:
x = 10
y = 3
addition = x + y  # 13
subtraction = x - y  # 7
multiplication = x * y  # 30
division = x / y  # 3.333...
floor_division = x // y  # 3
modulus = x % y  # 1

# 2. Comparison Operators: (conditions)
x = 5
y = 10
equal = x == y  # False
not_equal = x != y  # True
less_than = x < y  # True
greater_than = x > y  # False

# 3. Logical Operators:
x = 5
y = 10
z = 15
result = (x < y) and (y < z)  # True
result = (x < y) or (y < z)  # True


# 4. Identity Operators:
a = [1, 2, 3]
b = a
c = [1, 2, 3]
is_same = a is b  # True
is_not_same = a is not c  # True

# 5. Membership Operators:
my_list = [1, 2, 3, 4]
is_in_list = 3 in my_list  # True
is_not_in_list = 5 not in my_list  # True

# 6. Chained conditionals, also known as nested conditionals.
'''
    Chained conditionals.
            - a fundamental building block for creating more advanced algorithms and branching logic.
            - allow you to create complex decision-making structures.
            
    The practice of using multiple if, elif (else if), and else statements in succession 
    to evaluate a series of conditions one by one.
'''

x = 10

if x < 0:
    print("x is negative")
elif x == 0:
    print("x is zero")
else:
    print("x is positive")

