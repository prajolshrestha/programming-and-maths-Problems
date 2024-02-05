import argparse

def main():
    parser = argparse.ArgumentParser(description="A simple python script to demonstrate cmd-line argument parsing.")


    # Adding cmd-line arguments
    parser.add_argument('--name', type=str, default='World', help='Specify a name to greet (default: World)')
    parser.add_argument('--repeat', type=int, default=1, help='Specify the number of times to repeat the greeting (default: 1)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')

    args = parser.parse_args()

    # Greet the user
    greeting = f'Hello, {args.name}!'
    if args.verbose:
        greeting += ' (Verbose mode enabled)'
    print(greeting)

    # Repeat the greeting as per the specified count
    for _ in range(args.repeat - 1):
        print(greeting)

if __name__ == '__main__':
    main()