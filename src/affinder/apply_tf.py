from magicgui import magic_factory
from napari.layers import Layer
from napari.types import LayerDataTuple
from skimage import transform
import numpy as np


@magic_factory
def apply_affine(
        reference_layer: Layer, moving_layer: Layer
        ) -> LayerDataTuple:
    """Apply current affine transformation to selected layer.

    The input layer data will be transformed to match the reference layer.

    Parameters
    ----------
    reference_layer : napari.layers.Layer
        Layer to use as reference for affine transformation.
    moving_layer : napari.layers.Layer
        Layer to apply affine to.

    Returns
    -------
    LayerDataTuple
        Layer data tuple with transformed data.
    """
    # Get image data to be transformed
    im = moving_layer.data
    # Apply transformation
    if 'Image' in str(type(moving_layer)):
        affine = moving_layer.affine.affine_matrix
        if im.ndim == 2:
            affine = tform_matrix_rc2xy(affine)
        im = transform.warp(
                im,
                np.linalg.inv(affine),
                order=0,
                output_shape=reference_layer.data.shape,
                preserve_range=True
                )
        layertype = 'image'
        ref_metadata = {
                n: getattr(reference_layer, n)
                for n in ['scale', 'translate', 'rotate', 'shear']
                }
        mov_metadata = moving_layer.as_layer_data_tuple()[1]
        name = {'name': moving_layer.name + '_transformed'}

        metadata = {**mov_metadata, **ref_metadata, **name}
    else:
        raise NotImplementedError(
                'Only image transforms supported at this point.'
                )

    return (im, metadata, layertype)


def tform_matrix_rc2xy(affine_matrix: np.ndarray):
    """
    Convert an affine transformation matrix from the (unusual) convention where rows represent x-coordinates and columns
    represent y-coordinates to the more standard convention where rows represent y-coordinates and columns represent
    x-coordinates. This is necessary when working with libraries like sci-kit image that use the unconventional
    row= x, column= y convention.

    Parameters:
    -----------
        affine_matrix (numpy.ndarray): A 3x3 affine transformation matrix.

    Returns:
        numpy.ndarray: The transformed affine matrix with rows representing y-coordinates and columns representing
                       x-coordinates.
    """
    # swap columns
    new_affine_matrix = affine_matrix[:, [1, 0, 2]]
    # swap rows
    new_affine_matrix = new_affine_matrix[[1, 0, 2], :]
    return new_affine_matrix
