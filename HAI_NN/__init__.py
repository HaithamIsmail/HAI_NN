# __init__.py

VERSION = (1, 0, 0)

__author__ = 'Haitham Ismail'
__email__ = 'haithamismail.hi@gmail.com'
__version__ = '.'.join(map(str, VERSION))
__description__ = 'A neural network framework using Numpy package'

__all__ = [
    'Layers',
    'Activations',
    'Losses',
    'Optimizers'
]