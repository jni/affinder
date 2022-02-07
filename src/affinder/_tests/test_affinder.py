from affinder import start_affinder
from affinder.affinder import AffineTransformChoices
from skimage import data, transform
import numpy as np
from itertools import product
import zarr
import napari
import pytest

layer0_pts = np.array([[140.38371886,
                        322.5390704], [181.91866481, 319.65803368],
                       [176.15659138, 259.1562627],
                       [140.14363246, 254.59462124]])
layer1_pts = np.array([[70.94741072,
                        117.37477536], [95.80911919, 152.00358359],
                       [143.16475439, 118.55866623],
                       [131.32584559, 83.33791256]])

# get reference and moving layer types
im0 = data.camera()
im1 = transform.rotate(im0[100:, 32:496], 60)
labels0 = zarr.open('./src/affinder/_tests/labels0.zarr', mode='r')
labels1 = zarr.open('./src/affinder/_tests/labels1.zarr', mode='r')


def make_vector_border(layer_pts):
    vectors = np.zeros((layer_pts.shape[0], 2, layer_pts.shape[1]))
    for n in range(layer_pts.shape[0]):
        vectors[n, 0, :] = layer_pts[n, :]
        vectors[n, 1, :] = layer_pts[(n+1)
                                     % layer_pts.shape[0], :] - layer_pts[n, :]
    return vectors


vectors0 = make_vector_border(layer0_pts)
vectors1 = make_vector_border(layer1_pts)

ref = [
        napari.layers.Image(im0),
        napari.layers.Shapes(layer0_pts),
        napari.layers.Points(layer0_pts),
        napari.layers.Labels(labels0),
        napari.layers.Vectors(vectors0),
        ]
mov = [
        napari.layers.Image(im1),
        napari.layers.Shapes(layer1_pts),
        napari.layers.Points(layer1_pts),
        napari.layers.Labels(labels1),
        napari.layers.Vectors(vectors1),
        ]
# TODO add tracks layer types, after multidim affine support added


@pytest.mark.parametrize("reference,moving", [p for p in product(ref, mov)])
def test_layer_types(make_napari_viewer, tmp_path, reference, moving):

    viewer = make_napari_viewer()

    l0 = viewer.add_layer(reference)
    viewer.layers[-1].name = "layer0"
    viewer.layers[-1].colormap = "green"

    l1 = viewer.add_layer(moving)
    viewer.layers[-1].name = "layer1"
    viewer.layers[-1].colormap = "magenta"

    my_widget_factory = start_affinder()
    my_widget_factory(
            viewer=viewer,
            reference=l0,
            moving=l1,
            model=AffineTransformChoices.affine,
            output=tmp_path / 'my_affine.txt'
            )

    viewer.layers['layer0_pts'].data = layer0_pts
    viewer.layers['layer1_pts'].data = layer1_pts

    actual_affine = np.asarray(l1.affine)
    expected_affine = np.array([[0.48155037, 0.85804854, 5.43577937],
                                [-0.88088632, 0.49188026, 328.20642821],
                                [0., 0., 1.]])

    np.testing.assert_allclose(actual_affine, expected_affine)
