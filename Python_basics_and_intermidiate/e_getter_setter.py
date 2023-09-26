class getter_setter:
    def __init__(self) -> None:
        self._value = 0 # Private variable
    
    @property
    def valuee(self):
        return self._value
    
    @valuee.setter
    def valuee(self, new_value):
        self._value = new_value


obj = getter_setter()

# Use the property 'getter' to retrieve the current value
current_value = obj.valuee
print(f'Getting current value: {current_value}')

# Use the property 'setter' to update the value
obj.valuee = 42

updated_value = obj.valuee
print(f'Updated value: {updated_value}')