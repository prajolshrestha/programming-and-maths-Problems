"""
    Lets add constraint: temprature of any object cannot reach below -273.15 degrees celsius.

    How??

    Solution: Hide the attribute temprature (make it private) and 
              define new getter and setter methods to manipulate it.

    How?
            modify the code:
                obj.temperature ==> obj.get_temperature()
                obj.temperature = value ==> obj.set_temperature(value) 

    Problem: This refactoring can cause problem while dealing with hundreds of thousands of lines of codes.
             - not backwards compatible
    
    Solution: @property
"""

class Celsius:
    def __init__(self, temperature = 0):
        self.set_temperature(temperature)
    
    def to_fahrenheit(self):
        return (self.get_temperature() * 1.8) + 32
    
    # getter method
    def get_temperature(self):
        return self._temperature # private variable

    # setter method
    def set_temperature(self, value):
        # Restriction implemented
        if value < -273.15:
            raise ValueError("Temperatue below -273.15 is not possible")
        self._temperature = value

#temp = input('Enter a temperature')
human = Celsius(37) # create new object, set_temperature is internally called by __init__
print(human.get_temperature()) # get temperatue attribute via a getter
print(human.to_fahrenheit()) # get to_fahrenheit method, get_temperatue() called by the method itself
    
human = Celsius(-300) 
print(human.get_temperature())
print(human.to_fahrenheit()) 