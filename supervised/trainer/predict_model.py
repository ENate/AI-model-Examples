""" To print version"""
import tensorflow as tf


def print_version():
    """Print versions"""
    print(tf.__version__)
    
if __name__ == "__main__":
    print_version()