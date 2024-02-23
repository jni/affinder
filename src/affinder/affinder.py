import functools
import pathlib
import warnings
from enum import Enum
from typing import Optional
import napari
from napari.layers import Image, Labels, Shapes, Points, Vectors
from enum import Enum
import pathlib
from copy import deepcopy

import numpy as np
import toolz as tz
from magicgui import magic_factory
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
    if len(viewer.dims.displayed) == layer.extent.world.shape[1]:
        extent = layer.extent.world
    else:
        extent = layer.extent.world[:, viewer.dims.displayed]
    size = extent[1] - extent[0]
    center = extent[0] + size/2
    viewer.camera.center = center
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        canvas_size = viewer._canvas_size
    viewer.camera.zoom = np.min(canvas_size) / np.max(size)


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
        output
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
                mat = calculate_transform(
                        pts0, pts1, ndim, model_class=model_class
                        )
                ref_mat = reference_image_layer.affine.affine_matrix
                # must shrink ndims of affine matrix if dims of image layer is bigger than moving layer #####
                if reference_image_layer.ndim > moving_image_layer.ndim:
                    ref_mat = convert_affine_to_ndims(
                            ref_mat, moving_image_layer.ndim
                            )
                # must pad affine matrix with identity matrix if dims of moving layer smaller #####
                moving_image_layer.affine = convert_affine_to_ndims(
                        (ref_mat @ mat.params), moving_image_layer.ndim
                        )
                moving_points_layer.affine = convert_affine_to_ndims(
                        (ref_mat @ mat.params), moving_image_layer.ndim
                        )
                if output is not None:
                    np.savetxt(output, np.asarray(mat.params), delimiter=',')
            viewer.layers.selection.active = reference_points_layer
            reference_points_layer.mode = 'add'
            viewer.layers.move(viewer.layers.index(reference_image_layer), -1)
            viewer.layers.move(viewer.layers.index(reference_points_layer), -1)
            reset_view(viewer, reference_image_layer)


def close_affinder(layers, callback):
    for layer in layers:
        layer.events.data.disconnect(callback)
        layer.mode = 'pan_zoom'


# make function to remove points layers after finishing
def remove_pts_layers(viewer, layers):
    for layer in layers:
        viewer.layers.remove(layer)


def convert_affine_to_ndims(affine, target_ndim):
    """Either embed or slice an affine matrix to match the target ndims."""
    affine_matrix = np.asarray(affine)
    diff = np.shape(affine_matrix)[0] - 1 - target_ndim
    if diff == 0:
        out = affine_matrix
    elif diff < 0:
        # target is larger, so embed
        out = np.identity(target_ndim + 1)
        out[-diff:, -diff:] = affine_matrix
    else:  # diff > 0
        out = affine_matrix[diff:, diff:]

    return out


def _update_unique_choices(widget, choice_name):
    """Update the selected choice in a ComboBox widget to be unique.

    When `choice_name` is selected by another widget, and the choice in
    `widget` needs to be different, this callback can be called to update the
    choice in `widget`.
    """
    if not isinstance(choice_name, str):
        # in some circumstances, widget.changed.connect passes the choice
        # name to the callback, and in other cases it's the actual choice
        # value. Here we coerce it to always be the name but that's an
        # arbitrary choice.
        choice_name = choice_name.name
    choices = widget.choices
    choice_names = [value.name for value in choices]
    index = choice_names.index(choice_name)
    value = widget.choices[index]
    if widget.value is value:
        next_index = (index+1) % len(choices)
        with widget.changed.blocked():
            widget.value = widget.choices[next_index]


def _on_affinder_main_init(widget):
    """Make sure that the reference and moving image are not the same."""
    widget.reference.changed.connect(
            lambda v: _update_unique_choices(widget.moving, v)
            )
    widget.moving.changed.connect(
            lambda v: _update_unique_choices(widget.reference, v)
            )
    _update_unique_choices(widget.moving, widget.reference.current_choice)


@magic_factory(
        widget_init=_on_affinder_main_init,
        call_button='Start',
        layout='vertical',
        output={
                'mode': 'w', 'label': 'Save transformation as', 'filter':
                        '*.txt'
                },
        viewer={'visible': False, 'label': ' '},
        delete_pts={
                'label':
                        'Delete points layers when done',
                'tooltip': (
                        'If ticked, the points layers used in alignment '
                        'will be deleted when clicking "Finish".'
                        ),
                },
        )
def start_affinder(
        viewer: 'napari.viewer.Viewer',
        *,
        reference: 'napari.layers.Layer',
        reference_points: Optional['napari.layers.Points'] = None,
        moving: 'napari.layers.Layer',
        moving_points: Optional['napari.layers.Points'] = None,
        model: AffineTransformChoices,
        output: Optional[pathlib.Path] = None,
        delete_pts: bool = False,
        ):
    mode = start_affinder._call_button.text  # can be "Start" or "Finish"

    if mode == 'Start':

        # focus on the reference layer
        reset_view(viewer, reference)
        # set points layer for each image
        points_layers = [reference_points, moving_points]
        # Use C0 and C1 from matplotlib color cycle
        points_layers_to_add = [(reference, (0.122, 0.467, 0.706, 1.0)),
                                (moving, (1.0, 0.498, 0.055, 1.0))]

        # make points layer if it was not specified
        estimation_ndim = min(reference.ndim, moving.ndim)
        for i in range(len(points_layers)):
            if points_layers[i] is None:
                layer, color = points_layers_to_add[i]
                new_layer = viewer.add_points(
                        ndim=estimation_ndim, # ndims of all points layers same lowest ndim of reference or moving
                        name=layer.name + '_pts',
                        affine=convert_affine_to_ndims(
                                layer.affine, estimation_ndim
                                ),
                        face_color=[color],
                        )
                points_layers[i] = new_layer
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
                output=output
                )
        pts_layer0.events.data.connect(callback)
        pts_layer1.events.data.connect(callback)

        # get the layer order started
        for layer in [moving, pts_layer1, reference, pts_layer0]:
            viewer.layers.move(viewer.layers.index(layer), -1)

        viewer.layers.selection.active = pts_layer0
        pts_layer0.mode = 'add'

        start_affinder.close = functools.partial(
                close_affinder, points_layers, callback
                )
        start_affinder.remove_points_layers = functools.partial(
                remove_pts_layers, viewer, points_layers
                )
        # change the button/mode for next run
        start_affinder._call_button.text = 'Finish'
    else:  # we are in Finish mode
        start_affinder.close()
        if delete_pts:
            start_affinder.remove_points_layers()
        start_affinder._call_button.text = 'Start'


def calculate_transform(src, dst, ndim, model_class=AffineTransform):
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
    # convert points to correct dimension (from right bottom corner)
    # pos_val = lambda x: x if x > 0 else 0

    # do transform
    model = model_class(dimensionality=ndim)
    model.estimate(dst, src)  # we want
    # the inverse
    return model
