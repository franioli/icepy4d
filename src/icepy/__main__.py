import icepy
import importlib

# Try importing ICEpy4D module.


def main():
    try:
        importlib.import_module("icepy")
    except:
        raise ImportError("Unable to import ICEpy4D module")


if __name__ == "__main__":
    main()
