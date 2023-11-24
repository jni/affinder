from pathlib import Path

import numpy as np
from magicgui import magic_factory


@magic_factory(call_button='Load', affine={'mode': 'r'})
def load_affine(layer: 'napari.layers.Layer', affine: Path):
    """Load affine matrix from a file.

    Parameters
    ----------
    layer : napari.layers.Layer
        Layer to load affine to.
    affine : string or path
        Path to the file containing affine matrix. Must be
        comma-delimited txt.
    """
    affine = np.loadtxt(affine, delimiter=',')
    layer.affine = affine
