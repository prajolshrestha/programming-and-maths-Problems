"""
    Property with decorator

    We reuse temperature name while defining our getter and setter functions.

"""

class Celsius:
    def __init__(self, temperature=0):
        self.temperature = temperature
    
    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32
    
    @property
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
