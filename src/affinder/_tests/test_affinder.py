from affinder import start_affinder, copy_affine, apply_affine, load_affine
from affinder.affinder import AffineTransformChoices
from skimage import data, transform
import numpy as np
from itertools import product
import zarr
import napari
import pytest
from pathlib import Path
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

this_dir = Path(__file__).parent.absolute()
nuclei3D_2Dpts = nuclei3D_pts[:, 1:]
nuclei2D_2Dpts = nuclei2D_3Dpts[:, 1:]
nuclei2D_transformed_2Dpts = nuclei2D_transformed_3Dpts[:, 1:]

# get reference and moving layer types
nuclei2D = data.cells3d()[30, 1, :, :]  # (256, 256)
nuclei2D_transformed = transform.rotate(
        nuclei2D[10:, 32:496], 60
        )  # (246, 224)
nuclei3d = data.cells3d()[:, 1, :, :]  # (60, 256, 256)

nuclei2d_labels = zarr.open(this_dir / 'nuclei2D_labels.zarr', mode='r')
nuclei2d_labels_transformed = zarr.open(
        this_dir / 'nuclei2D_transformed_labels.zarr', mode='r'
        )
nuclei3d_labels = zarr.open(this_dir / 'nuclei3D_labels.zarr', mode='r')

im0 = data.camera()
im1 = transform.rotate(im0[100:, 32:496], 60)
labels0 = zarr.open(this_dir / 'labels0.zarr', mode='r')
labels1 = zarr.open(this_dir / 'labels1.zarr', mode='r')


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


nuc2D = generate_all_layer_types(nuclei2D, nuclei2D_2Dpts, nuclei2d_labels)
nuc2D_t = generate_all_layer_types(
        nuclei2D_transformed, nuclei2D_transformed_2Dpts,
        nuclei2d_labels_transformed
        )
nuc3D = generate_all_layer_types(nuclei3d, nuclei3D_pts, nuclei3d_labels)


# 2D as reference, 2D as moving
@pytest.mark.parametrize(
        "reference,moving,model_class", [
                p for p in
                product(nuc2D, nuc2D_t, [t for t in AffineTransformChoices])
                ]
        )
def test_2D_2D(make_napari_viewer, tmp_path, reference, moving, model_class):

    viewer = make_napari_viewer()

    l0 = viewer.add_layer(reference)
    l0.name = "layer0"

    l1 = viewer.add_layer(moving)
    l1.name = "layer1"

    affinder_widget = start_affinder()
    affinder_widget(
            viewer=viewer,
            reference=l0,
            moving=l1,
            model=model_class,
            output=tmp_path / 'my_affine.txt'
            )

    viewer.layers['layer0_pts'].data = nuclei2D_2Dpts
    viewer.layers['layer1_pts'].data = nuclei2D_transformed_2Dpts

    actual_affine = np.asarray(l1.affine)

    model = model_class.value(dimensionality=2)
    model.estimate(
            viewer.layers['layer1_pts'].data, viewer.layers['layer0_pts'].data
            )
    expected_affine = model.params

    np.testing.assert_allclose(
            actual_affine, expected_affine, rtol=10, atol=1e-10
            )


# 3D as reference, 2D as moving
@pytest.mark.parametrize(
        "reference,moving,model_class", [
                p for p in
                product(nuc3D, nuc2D_t, [t for t in AffineTransformChoices])
                ]
        )
def test_3D_2D(make_napari_viewer, tmp_path, reference, moving, model_class):

    viewer = make_napari_viewer()

    l0 = viewer.add_layer(reference)
    l0.name = "layer0"

    l1 = viewer.add_layer(moving)
    l1.name = "layer1"

    affinder_widget = start_affinder()
    affinder_widget(
            viewer=viewer,
            reference=l0,
            moving=l1,
            model=model_class,
            output=tmp_path / 'my_affine.txt'
            )

    viewer.layers['layer0_pts'].data = nuclei3D_2Dpts
    viewer.layers['layer1_pts'].data = nuclei2D_transformed_2Dpts

    actual_affine = np.asarray(l1.affine)

    model = model_class.value(dimensionality=2)
    model.estimate(
            viewer.layers['layer1_pts'].data, viewer.layers['layer0_pts'].data
            )
    expected_affine = model.params

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


def test_apply_affine_with_scale():
    ref_im = np.random.random((5, 5))
    mov_im = ndi.zoom(ref_im, 2, order=0)

    ref_layer = napari.layers.Image(ref_im, scale=(0.2, 0.2))
    mov_layer = napari.layers.Image(mov_im, scale=(0.4, 0.4))
    mov_layer.affine = np.array([[0.25, 0, 0], [0, 0.25, 0], [0, 0, 1]])

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

    np.testing.assert_allclose(layer.affine, affine)


@pytest.mark.parametrize('remove_pts', [True, False])
def test_remove_points_layers(remove_pts, make_napari_viewer):
    """Check whether remove_points_layer option actually removes the layers."""
    ref_im = np.random.random((5, 5))
    mov_im = np.random.random((5, 5))
    ref_pts = np.array([[1, 1], [2, 2], [1, 4]], dtype=float)
    mov_pts = np.array([[4, 1], [2, 2], [1, 4]], dtype=float)

    viewer = make_napari_viewer()
    ref_layer = viewer.add_image(ref_im)
    mov_layer = viewer.add_image(mov_im)
    qtwidget, widget = viewer.window.add_plugin_dock_widget(
            'affinder', 'Start affinder'
            )
    widget(
            viewer=viewer,
            reference=ref_layer,
            moving=mov_layer,
            model=AffineTransformChoices.affine,
            )
    viewer.layers['ref_im_pts'].data = ref_pts
    viewer.layers['mov_im_pts'].data = mov_pts

    widget(delete_pts=remove_pts)  # close the widget

    assert widget._call_button.text == 'Start'

    assert remove_pts != any(
            pt_layer in viewer.layers
            for pt_layer in ['ref_im_pts', 'mov_im_pts']
            )
