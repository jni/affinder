from affinder import start_affinder, copy_affine, apply_affine, load_affine
from affinder.affinder import AffineTransformChoices
from skimage import data, transform
import numpy as np
from itertools import product
import zarr
import napari
import pytest
from scipy import ndimage as ndi


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


def test_ensure_different_layers(make_napari_viewer):
    viewer = make_napari_viewer()
    image0 = np.random.random((50, 50))
    image1 = np.random.random((30, 30))
    image2 = np.random.random((40, 40))
    for image in [image0, image1, image2]:
        viewer.add_image(image)
    qtwidget, widget = viewer.window.add_plugin_dock_widget(
            'affinder', 'Start affinder'
            )
    assert widget.reference.value != widget.moving.value
    widget.reference.value = widget.moving.value
    assert widget.reference.value != widget.moving.value


def test_copy_affine():
    layer0 = napari.layers.Image(np.random.random((5, 5)))
    layer1 = napari.layers.Image(np.random.random((5, 5)))
    layer0.affine = np.array([[0.9, 0.1, 5], [0.4, 0.2, 9], [0, 0, 1]])

    widget = copy_affine()
    widget(layer0, layer1)
    np.testing.assert_allclose(layer0.affine, layer1.affine)


def test_apply_affine():
    ref_im = np.random.random((5, 5))
    mov_im = ndi.zoom(ref_im, 2, order=0)

    ref_layer = napari.layers.Image(ref_im)
    mov_layer = napari.layers.Image(mov_im)
    mov_layer.affine = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])

    widget = apply_affine()
    res_layer = widget(ref_layer, mov_layer)

    np.testing.assert_allclose(res_layer[0], ref_im)


def test_apply_affine_nonimage():
    ref_im = np.random.random((5, 5))
    mov_pts = np.random.random((5, 2))

    ref_layer = napari.layers.Image(ref_im)
    mov_layer = napari.layers.Points(mov_pts)
    mov_layer.affine = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])

    widget = apply_affine()
    with pytest.raises(NotImplementedError):
        widget(ref_layer, mov_layer)


def test_load_affine():
    existing_affine = np.array([[0.9, 0.1, 5], [0.4, 0.2, 9], [0, 0, 1]])

    layer = napari.layers.Image(np.random.random((5, 5)))
    layer.affine = existing_affine

    widget = load_affine(,
    widget(layer, './src/affinder/_tests/load_test/2d_test_affine.txt')
    np.testing.assert_allclose(
            layer.affine,
            np.loadtxt('./src/affinder/_tests/load_test/2d_test_affine.txt',
                       delimiter=',') @ existing_affine
            )

