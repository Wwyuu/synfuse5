import importlib


def import_algorithm(name):
    """Load Net from research2v1.models.<name>."""
    return importlib.import_module('research2v1.models.' + name.lower())
