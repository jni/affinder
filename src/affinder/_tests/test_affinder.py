from affinder import start_affinder, copy_affine, apply_affine, load_affine
from affinder.affinder import AffineTransformChoices
from skimage import data, transform
import numpy as np
from itertools import product
import zarr
import napari
import pytest
from copy import copy
from scipy import ndimage as ndi

nuclei3D_pts = np.array([[30., 68.47649186, 67.08770344],
                         [30., 85.14195298, 51.81103074],
                         [30., 104.58499096, 43.94122966],
                         [30., 154.58137432, 113.38065099]])

nuclei2D_3Dpts = np.array([[0, 68.47649186, 67.08770344],
                           [0, 85.14195298, 51.81103074],
                           [0, 104.58499096, 43.94122966],
                           [0, 154.58137432, 113.38065099]])

nuclei2D_transformed_3Dpts = np.array([[0, 154.44736842, 18.95499262],
                                       [0, 176.10600098, 24.49557304],
                                       [0, 195.2461879, 35.57673389],
                                       [0, 160.49163797, 116.67068372]])

nuclei3D_2Dpts = nuclei3D_pts[:, 1:]
nuclei2D_2Dpts = nuclei2D_3Dpts[:, 1:]
nuclei2D_transformed_2Dpts = nuclei2D_transformed_3Dpts[:, 1:]

# get reference and moving layer types
nuclei2D = data.cells3d()[30, 1, :, :]  # (256, 256)
nuclei2D_transformed = transform.rotate(
        nuclei2D[10:, 32:496], 60
        )  # (246, 224)
nuclei3D = data.cells3d()[:, 1, :, :]  # (60, 256, 256)

nuclei2D_labels = zarr.open(
        './src/affinder/_tests/nuclei2D_labels.zarr', mode='r'
        )  #########
nuclei2D_transformed_labels = zarr.open(
        './src/affinder/_tests/nuclei2D_transformed_labels.zarr', mode='r'
        )  #########
nuclei3D_labels = zarr.open(
        './src/affinder/_tests/nuclei3D_labels.zarr', mode='r'
        )  #########


def make_vector_border(layer_pts):
    vectors = np.zeros((layer_pts.shape[0], 2, layer_pts.shape[1]))
    for n in range(layer_pts.shape[0]):
        vectors[n, 0, :] = layer_pts[n, :]
        vectors[n, 1, :] = layer_pts[(n+1)
                                     % layer_pts.shape[0], :] - layer_pts[n, :]
    return vectors


def generate_all_layer_types(image, pts, labels):
    layers = [
            napari.layers.Image(image),
            napari.layers.Shapes(pts),
            napari.layers.Points(pts),
            napari.layers.Labels(labels),
            napari.layers.Vectors(make_vector_border(pts)),
            ]

    return layers


nuc2D = generate_all_layer_types(nuclei2D, nuclei2D_2Dpts, nuclei2D_labels)
nuc2D_t = generate_all_layer_types(
        nuclei2D_transformed, nuclei2D_transformed_2Dpts,
        nuclei2D_transformed_labels
        )
nuc3D = generate_all_layer_types(nuclei3D, nuclei3D_pts, nuclei3D_labels)

################
################
################


# 2D as reference, 2D as moving
@pytest.mark.parametrize(
        "reference,moving", [p for p in product(nuc2D, nuc2D_t)]
        )
def test_2D_2D(make_napari_viewer, tmp_path, reference, moving):

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

    viewer.layers['layer0_pts'].data = nuclei2D_2Dpts
    viewer.layers['layer1_pts'].data = nuclei2D_transformed_2Dpts

    actual_affine = np.asarray(viewer.layers['layer1'].affine)
    expected_affine = np.array([[0.54048889, 0.8468468, -30.9685414],
                                [-0.78297398, 0.52668962, 177.6241674],
                                [0., 0., 1.]])

    np.testing.assert_allclose(
            actual_affine, expected_affine, rtol=10, atol=1e-10
            )


# 3D as reference, 2D as moving
@pytest.mark.parametrize(
        "reference,moving", [p for p in product(nuc3D, nuc2D)]
        )
def test_3D_2D(make_napari_viewer, tmp_path, reference, moving):

    viewer = make_napari_viewer()

    l0 = viewer.add_layer(reference)
    viewer.layers[-1].name = "layer0"
    viewer.layers[-1].colormap = "green"

    # affinder currently changes the moving layer data when dims are different
    # so need to copy
    l1 = viewer.add_layer(copy(moving))
    viewer.layers[-1].name = "layer1"
    viewer.layers[-1].colormap = "magenta"

    my_widget_factory = start_affinder()
    my_widget_factory(
            viewer=viewer,
            reference=l0,
            moving=l1,
            model=AffineTransformChoices.Euclidean,
            output=tmp_path / 'my_affine.txt'
            )

    viewer.layers['layer0_pts'].data = nuclei3D_pts
    viewer.layers['layer1_pts'].data = nuclei2D_3Dpts

    actual_affine = np.asarray(viewer.layers['layer1'].affine)
    # start_affinder currently makes a clone of moving layer when it's of
    # type Points of Vectors and not same dimensions as reference layer - so l1
    # is a redundant layer that is no longer used as the real moving layer -
    # this is why we use viewer.layers['layer1] instead of l1

    expected_affine = np.array(
            [[1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.00000000e+01],
             [0.00000000e+00, 1.00000000e+00, 2.89023467e-17, 0.00000000e+00],
             [0.00000000e+00, -7.90288925e-18, 1.00000000e+00, 1.42108547e-14],
             [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
            )

    np.testing.assert_allclose(
            actual_affine, expected_affine, rtol=10, atol=1e-10
            )


# 2D as reference, 3D as moving
@pytest.mark.parametrize(
        "reference,moving", [p for p in product(nuc2D, nuc3D)]
        )
def test_2D_3D(make_napari_viewer, tmp_path, reference, moving):

    viewer = make_napari_viewer()

    l0 = viewer.add_layer(reference)
    viewer.layers[-1].name = "layer0"
    viewer.layers[-1].colormap = "green"

    # affinder currently changes the moving layer data when dims are different
    # so need to copy
    l1 = viewer.add_layer(copy(moving))
    viewer.layers[-1].name = "layer1"
    viewer.layers[-1].colormap = "magenta"

    my_widget_factory = start_affinder()
    my_widget_factory(
            viewer=viewer,
            reference=l0,
            moving=l1,
            model=AffineTransformChoices.Euclidean,
            output=tmp_path / 'my_affine.txt'
            )

    viewer.layers['layer0_pts'].data = nuclei2D_2Dpts
    viewer.layers['layer1_pts'].data = nuclei3D_2Dpts

    actual_affine = np.asarray(viewer.layers['layer1'].affine)
    # start_affinder currently makes a clone of moving layer when it's of
    # type Points of Vectors and not same dimensions as reference layer - so l1
    # is a redundant layer that is no longer used as the real moving layer -
    # this is why we use viewer.layers['layer1] instead of l1
    expected_affine = np.array([[1.00000000e+00,  0.00000000e+00,  0.00000000e+00, 0.00000000e+00],
                                [0.00000000e+00, 1.000000e+00, 2.890235e-17, 0.000000e+00],
                                [0.00000000e+00, -7.902889e-18, 1.000000e+00, 1.421085e-14],
                                [0.00000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]])

    np.testing.assert_allclose(
            actual_affine, expected_affine, rtol=10, atol=1e-10
            )


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


def test_load_affine(tmp_path):
    affile = tmp_path / 'test_affine.txt'
    affine = np.array([[2, 0, 5], [0, 2, 5], [0, 0, 1]])
    np.savetxt(affile, affine, delimiter=',')

    layer = napari.layers.Image(np.random.random((5, 5)))

    widget = load_affine()
    widget(layer, affile)

    np.testing.assert_allclose(
        layer.affine, affine
    )

