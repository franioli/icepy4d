import icepy4d
import importlib

# Try importing ICEpy4D module.


def main():
    try:
        importlib.import_module("icepy4d")
    except:
        raise ImportError("Unable to import ICEpy4D module")


if __name__ == "__main__":
    main()
