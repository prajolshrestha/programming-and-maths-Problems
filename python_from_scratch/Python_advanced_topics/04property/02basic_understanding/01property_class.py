""" 
    Q. What does property does?

    Setter: 
            self.temperature = temperature, automically calls set_temperature()
            
    Getter:
            Any access like c.temperature, automatically calls get_temperature()

    Take away: We can see no modification of class, is required in the implementation of the value constraint.
               Thus, our implementation is backward compatible.

    Note: Actual temperature value is stored in the private variable '_temperature'.
          The temperature attribute is a property object which provides an interface to this private variable.
"""

class Celsius:
    def __init__(self, temperature = 0):
        self.temperature = temperature # calls set_temperature 
    
    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32 # calls get_temperature
    
    # getter method
    def get_temperature(self):
        print("Getting value ...")
        return self._temperature # private variable

    # setter method
    def set_temperature(self, value):
        print("Setting value ...")
        # Restriction implemented
        if value < -273.15:
            raise ValueError("Temperatue below -273.15 is not possible")
        self._temperature = value
    
    # creating a property object
    temperature = property(get_temperature, set_temperature) # acts as interface to the private variable '_temperature'
    """
        temperature = property(get_temperature,set_temperature)

        can be broken down as:

        temperature = property()
        temperature = temperature.getter(get_temperature)
        temperature = temperature.setter(set_temparature)
    """

human = Celsius(37) # create a new object, calls set_temperature()
print(human.temperature) # calls get_temperatue() instead of a dictionary __dict__ look-up
print(human.to_fahrenheit())

human.temperature = -300 # calls set_temperature()