"""
    Class without getter and setter:
        - stores temperature in degree celsius.
        - has a method to convert the temperature into degrees Fahrenheit.
"""
class Celsius:
    def __init__(self, temprature = 0):
        self.temprature = temprature
    
    def to_fahrenheit(self):
        return (self.temprature * 1.8) + 32
    
# We can make objects out of the above class and manipulate the 'temprature' attribute as we wish

# Q. How to set and get attributes?

human = Celsius() # create a new object
human.temprature = 37 # set the temprature attribute
print(human.temprature) # get the temprature attribute
print(human.to_fahrenheit()) # get the to_fahrenheit method

print(human.__dict__)

"""
    output: 37
            98.60000000000001

    so, whenever we assign or retrive any object attribute like temperature,
        --> python search it in the object's built-in __dic__ dictionary attribute as
                human.__dict__  ==> {'temprature': 37}

    Take away: human.temprature internally becomes human.__dict__['temprature]
"""