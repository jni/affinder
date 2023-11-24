try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .affinder import start_affinder
from .copy_tf import copy_affine
from .load_tf import load_affine
from .apply_tf import apply_affine
