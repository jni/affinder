from magicgui import magic_factory
from skimage import transform
from napari.utils.transforms import CompositeAffine
import numpy as np


def _apply_affine_image(image, affine, order, reference_shape):
    """Apply affine transformation to image.

    Parameters
    ----------
    image : numpy.ndarray
        Image to be transformed.
    affine : numpy.ndarray
        Affine transformation matrix.
    order : int
        The order of the interpolation.
    reference_shape : tuple
        Shape of the output image.

    Returns
    -------
    numpy.ndarray
        Transformed image.
    """
    if image.ndim == 2:
        affine = tform_matrix_rc2xy(affine)
    return transform.warp(
            image,
            np.linalg.inv(affine),
            order=order,
            output_shape=reference_shape,
            preserve_range=True
            )


@magic_factory
def apply_affine(
        reference_layer: 'napari.layers.Layer',
        moving_layer: 'napari.layers.Layer',
        order: int = 0,
        ) -> 'napari.types.LayerDataTuple':
    """Apply current affine transformation to selected layer.

    The input layer data will be transformed to match the reference layer.

    Parameters
    ----------
    reference_layer : napari.layers.Layer
        Layer to use as reference for affine transformation.
    moving_layer : napari.layers.Layer
        Layer to apply affine to.
    order : int in {0, 1, 2, 3, 4, 5}
        The order of the interpolation.

    Returns
    -------
    LayerDataTuple
        Layer data tuple with transformed data.
    """
    if 'Image' not in str(type(moving_layer)):
        raise NotImplementedError(
                'Only image transforms supported at this point.'
                )

    reference_meta = CompositeAffine(
            scale=reference_layer.scale,
            translate=reference_layer.translate,
            rotate=reference_layer.rotate,
            shear=reference_layer.shear,
            )
    moving_meta = CompositeAffine(
            scale=moving_layer.scale,
            translate=moving_layer.translate,
            rotate=moving_layer.rotate,
            shear=moving_layer.shear,
            )
    # Find the transformation relative to the reference image
    affine = (
            np.linalg.inv(reference_meta)
            @ np.linalg.inv(reference_layer.affine) @ moving_layer.affine
            @ moving_meta
            )

    # Apply the transformation
    transformed = _apply_affine_image(
            moving_layer.data, affine, order, reference_layer.data.shape
            )

    # Set the metadata
    layertype = 'image'
    ref_metadata = {
            n: getattr(reference_layer, n)
            for n in ['scale', 'translate', 'rotate', 'shear', 'affine']
            }
    mov_metadata = moving_layer.as_layer_data_tuple()[1]
    name = {'name': moving_layer.name + '_transformed'}

    metadata = {**mov_metadata, **ref_metadata, **name}

    return (transformed, metadata, layertype)


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
