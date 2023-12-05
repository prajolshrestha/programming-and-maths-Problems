"""
    @Property decorator --> USED  to create read-only properties or to define a getter method for a class attribute.

    We reuse temperature name while defining our getter and setter functions.

    Intuition: It allows us to access the attribute like an ordinary attribute, 
               but behind the scenes, a method is called to compute the value.

    Take away: - for Encapsulation
               - allows you to add custom logic to attribute access without changing the way you interact with the class externally.

"""

class Celsius:
    def __init__(self, temperature=0):
        self.temperature = temperature # we can also make this attribute private
    
    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32
    
    @property # Read only properties
    def temperature(self):
        print("Getting value ...")
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        print("Setting value ...")
        if value < -273.15:
            raise ValueError("Temperature below -273.15 is not possible")
        self._temperature = value

human = Celsius(37) #set
print(human.temperature) #get
print(human.to_fahrenheit()) #get

#human.temperature = -300 #set
coldest_thing = Celsius(-300)
