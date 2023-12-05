def greeting(name):
    def hello():
        return f"Hello, {name}!"
    return hello

greet = greeting('Prajol')
print(greet())