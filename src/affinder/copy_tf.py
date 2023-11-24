from magicgui import magic_factory
from napari.layers import Layer


@magic_factory
def copy_affine(src_layer: Layer, dst_layer: Layer):
    """Copy affine from one layer to another.

    Parameters
    ----------
    src_layer : napari.layers.Layer
        Layer to copy affine from.
    dst_layer : napari.layers.Layer
        Layer to copy affine to.
    """
    dst_layer.affine = src_layer.affine
