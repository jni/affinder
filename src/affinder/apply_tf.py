from magicgui import magic_factory
from skimage import transform
import numpy as np


@magic_factory
def apply_affine(
        reference_layer: 'napari.layers.Layer',
        moving_layer: 'napari.layers.Layer'
        ) -> 'napari.types.LayerDataTuple':
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
    if 'Image' not in str(type(moving_layer)):
        raise NotImplementedError(
            'Only image transforms supported at this point.'
        )

    # Get image data to be transformed
    im = moving_layer.data
    # Apply transformation
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

    return (im, metadata, layertype)


def tform_matrix_rc2xy(affine_matrix: np.ndarray):
    """Transpose the first and second indices of an affine matrix.

    This makes the matrix match the (soon to be deprecated) skimage convention
    in which the first row matches the second axis of a NumPy array and
    vice-versa.

    Parameters
    ----------
    affine_matrix : numpy.ndarray (D+1, D+1)
        An affine transformation matrix.

    Returns:
    numpy.ndarray :
        The 'transposed' affine matrix.
    """
    # swap columns
    swapped_cols = affine_matrix[:, [1, 0, 2]]
    # swap rows
    swapped_both = swapped_cols[[1, 0, 2], :]
    return swapped_both
