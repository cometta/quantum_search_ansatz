# pylint: skip-file
class Singleton(type):
    """Helper class to create singleton"""

    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(type(cls), cls).__call__(*args, **kwargs)
        return cls._instances[cls]
