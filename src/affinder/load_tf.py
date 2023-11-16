from pathlib import Path

import numpy as np
from magicgui import magic_factory
from napari.layers import Layer


@magic_factory(call_button='Load', affine={'mode': 'r'})
def load_affine(layer: Layer, affine: Path):
    """Load affine from string.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer to load affine to.
    affine : str
        Affine to load.
    """
    affine = np.loadtxt(affine, delimiter=',')
    layer.affine = affine
