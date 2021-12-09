from typing import Optional

from enum import Enum
import pathlib
import toolz as tz
from magicgui import magicgui, magic_factory
import numpy as np
from skimage.transform import (
        AffineTransform,
        EuclideanTransform,
        SimilarityTransform,
        )


class AffineTransformChoices(Enum):
    affine = AffineTransform
    Euclidean = EuclideanTransform
    similarity = SimilarityTransform


def reset_view(viewer: 'napari.Viewer', layer: 'napari.layers.Layer'):
    if viewer.dims.ndisplay != 2:
        return
    extent = layer.extent.world[:, viewer.dims.displayed]
    size = extent[1] - extent[0]
    center = extent[0] + size/2
    viewer.camera.center = center
    viewer.camera.zoom = np.min(viewer._canvas_size) / np.max(size)


@tz.curry
def next_layer_callback(
        value,  # we ignore the arguments returned with the event -- we will
        *args,  # instead introspect the layer data and selection state
        viewer,
        reference_image_layer,
        reference_points_layer,
        moving_image_layer,
        moving_points_layer,
        model_class,
        output,
        ):
    pts0, pts1 = reference_points_layer.data, moving_points_layer.data
    n0, n1 = len(pts0), len(pts1)
    ndim = pts0.shape[1]
    if reference_points_layer in viewer.layers.selection:
        if n0 < ndim + 1:
            return
        if n0 == ndim + 1:
            reset_view(viewer, moving_image_layer)
        if n0 > n1:
            viewer.layers.selection.active = moving_points_layer
            viewer.layers.move(viewer.layers.index(moving_image_layer), -1)
            viewer.layers.move(viewer.layers.index(moving_points_layer), -1)
            moving_points_layer.mode = 'add'
    elif moving_points_layer in viewer.layers.selection:
        if n1 == n0:
            # we just added enough points:
            # estimate transform, go back to layer0
            if n0 > ndim:
                mat = calculate_transform(pts0, pts1, model_class=model_class)
                moving_image_layer.affine = (
                        reference_image_layer.affine.affine_matrix @ mat.params
                        )
                moving_points_layer.affine = (
                        reference_image_layer.affine.affine_matrix @ mat.params
                        )
                if output is not None:
                    np.savetxt(output, np.asarray(mat.params), delimiter=',')
            viewer.layers.selection.active = reference_points_layer
            reference_points_layer.mode = 'add'
            viewer.layers.move(viewer.layers.index(reference_image_layer), -1)
            viewer.layers.move(viewer.layers.index(reference_points_layer), -1)
            reset_view(viewer, reference_image_layer)


# make a bindable function to shut things down
@magicgui
def close_affinder(layers, callback):
    for layer in layers:
        layer.events.data.disconnect(callback)
        layer.mode = 'pan_zoom'


@magic_factory(
        call_button='Start',
        layout='vertical',
        output={'mode': 'w'},
        viewer={'visible': False, 'label': ' '},
        )
def start_affinder(
        viewer: 'napari.viewer.Viewer',
        reference: 'napari.layers.Image',
        moving: 'napari.layers.Image',
        model: AffineTransformChoices,
        output: Optional[pathlib.Path] = None,
        ):
    mode = start_affinder._call_button.text  # can be "Start" or "Finish"

    if mode == 'Start':
        # focus on the reference layer
        reset_view(viewer, reference)
        # make a points layer for each image
        points_layers = []
        # Use C0 and C1 from matplotlib color cycle
        for layer, color in [
                (reference, (0.122, 0.467, 0.706, 1.0)),
                (moving, (1.0, 0.498, 0.055, 1.0)),
                ]:
            new_layer = viewer.add_points(
                    ndim=layer.ndim,
                    name=layer.name + '_pts',
                    affine=layer.affine,
                    face_color=[color],
                    )
            points_layers.append(new_layer)
        pts_layer0 = points_layers[0]
        pts_layer1 = points_layers[1]

        # make a callback for points added
        callback = next_layer_callback(
                viewer=viewer,
                reference_image_layer=reference,
                reference_points_layer=pts_layer0,
                moving_image_layer=moving,
                moving_points_layer=pts_layer1,
                model_class=model.value,
                output=output,
                )
        pts_layer0.events.data.connect(callback)
        pts_layer1.events.data.connect(callback)

        # get the layer order started
        for layer in [moving, pts_layer1, reference, pts_layer0]:
            viewer.layers.move(viewer.layers.index(layer), -1)

        viewer.layers.selection.active = pts_layer0
        pts_layer0.mode = 'add'

        close_affinder.layers.bind(points_layers)
        close_affinder.callback.bind(callback)

        # change the button/mode for next run
        start_affinder._call_button.text = 'Finish'
    else:  # we are in Finish mode
        close_affinder()
        start_affinder._call_button.text = 'Start'


def calculate_transform(src, dst, model_class=AffineTransform):
    """Calculate transformation matrix from matched coordinate pairs.

    Parameters
    ----------
    src : ndarray
        Matched row, column coordinates from source image.
    dst : ndarray
        Matched row, column coordinates from destination image.
    model_class : scikit-image transformation class, optional.
        By default, model=AffineTransform().

    Returns
    -------
    transform
        scikit-image Transformation object
    """
    model = model_class()
    model.estimate(dst, src)  # we want the inverse
    return model
