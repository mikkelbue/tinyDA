__version__ = '0.9.1'

from .chain import *
from .link import *
from .distributions import *
from .proposal import *
from .diagnostics import *
from .utils import *

try:
    from .ray import *
except ModuleNotFoundError:
    print('Ray module not found. Multiprocessing features are not available')
    pass
