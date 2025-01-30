def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"


def main():
    name = input("Enter your name: ")
    print(greet(name))


if __name__ == "__main__":
    main()
