class Singleton:
    _instance = None

    @staticmethod
    def get_instance():
        if Singleton._instance is None:
            print("There exists no object. So, Creating a new instance!")
            Singleton._instance = Singleton()
        
        return Singleton._instance
    # OR
    @classmethod
    def instance(cls):
        if cls._instance is None:
            print("There exists no object. So, Creating a new instance!")
            cls._instance = cls.__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        if Singleton._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Singleton._instance = self
    
def main():
    # Create an object
    singleton = Singleton()

    # Get the singleton instance
    singleton1 = Singleton.get_instance()
    singleton2 = Singleton.get_instance()

    # Attempt to create new object of the Singleton Class
    try:
        singleton3 = Singleton() # There exists a singleton object. Hence, throws an exception!
    except Exception as e:
        print(e)

    # Get the singleton instance using instance()
    singleton4 = Singleton.instance()
    
    	
    print(f"Singleton 1: {singleton1}")
    print(f"Singleton 2: {singleton2}")
    print(f"Singleton 4: {singleton4}")
    print("done")

if __name__ == "__main__":
    main()