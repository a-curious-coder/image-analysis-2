'''Changes print format to how Callum likes it'''

_print = print

def print(text, tab = 0, **nargs):
    _print("[*]" + "\t" * (tab + 1) + str(text), **nargs)
