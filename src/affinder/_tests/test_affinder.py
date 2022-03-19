from affinder import start_affinder
from affinder.affinder import AffineTransformChoices
from skimage import data, transform
import numpy as np
from itertools import product
import zarr
import napari
import pytest

nuclei3D_pts = np.array([[ 30.        ,  68.47649186,  67.08770344],
                       [ 30.        ,  85.14195298,  51.81103074],
                       [ 30.        , 104.58499096,  43.94122966],
                       [ 30.        , 154.58137432, 113.38065099]])

nuclei2D_3Dpts = np.array([[0, 68.47649186,  67.08770344],
                           [0,  85.14195298,  51.81103074],
                           [0, 104.58499096,  43.94122966],
                           [0, 154.58137432, 113.38065099]])

nuclei2D_transformed_3Dpts = np.array([[0, 154.44736842,  18.95499262],
                                       [0, 176.10600098,  24.49557304],
                                       [0, 195.2461879 ,  35.57673389],
                                       [0, 160.49163797, 116.67068372]])
nuclei2D_2Dpts = nuclei2D_3Dpts[:,1:]
nuclei2D_transformed_2Dpts = nuclei2D_transformed_3Dpts[:,1:]

# get reference and moving layer types
nuclei2D = data.cells3d()[30,1,:,:] # (256, 256)
nuclei2D_transformed = transform.rotate(nuclei2D[10:, 32:496], 60) # (246, 224)
nuclei3D = data.cells3d()[:,1,:,:] # (60, 256, 256)

nuclei2D_labels = zarr.open(
    './src/affinder/_tests/nuclei2D_labels.zarr',
    mode='r')#########
nuclei2D_transformed_labels = zarr.open(
    './src/affinder/_tests/nuclei2D_transformed_labels.zarr',
    mode='r')#########
nuclei3D_labels = zarr.open(
    './src/affinder/_tests/nuclei3D_labels.zarr',
    mode='r')#########

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
        #napari.layers.Points(pts),
        napari.layers.Labels(labels),
        #napari.layers.Vectors(make_vector_border(pts)),
        ]

    return layers

nuc2D = generate_all_layer_types(nuclei2D, nuclei2D_2Dpts, nuclei2D_labels)
nuc2D_t = generate_all_layer_types(nuclei2D_transformed,
                                   nuclei2D_transformed_2Dpts,
                                   nuclei2D_transformed_labels)
nuc3D = generate_all_layer_types(nuclei3D, nuclei3D_pts, nuclei3D_labels)

################
################
################

# 2D as reference, 2D as moving
@pytest.mark.parametrize("reference,moving", [p for p in product(nuc2D,
                                                                 nuc2D_t)])
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

    actual_affine = np.asarray(l1.affine)
    expected_affine = np.array([[  0.54048889,   0.8468468 , -30.9685414 ],
       [ -0.78297398,   0.52668962, 177.6241674 ],
       [  0.        ,   0.        ,   1.        ]])

    np.testing.assert_allclose(actual_affine, expected_affine)


# 3D as reference, 2D as moving
@pytest.mark.parametrize("reference,moving", [p for p in product(nuc3D, nuc2D)])
def test_3D_2D(make_napari_viewer, tmp_path, reference, moving):

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
            model=AffineTransformChoices.Euclidean,
            output=tmp_path / 'my_affine.txt'
            )

    viewer.layers['layer0_pts'].data = nuclei3D_pts
    viewer.layers['layer1_pts'].data = nuclei2D_3Dpts

    actual_affine = np.asarray(l1.affine)
    expected_affine = np.array([
        [ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.00000000e+01],
     [ 0.00000000e+00,  1.00000000e+00,  2.89023467e-17,  0.00000000e+00],
     [ 0.00000000e+00, -7.90288925e-18,  1.00000000e+00,  1.42108547e-14],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    np.testing.assert_allclose(actual_affine, expected_affine)



# 2D as reference, 3D as moving
@pytest.mark.parametrize("reference,moving", [p for p in product(nuc2D,
                                                                 nuc3D)])
def test_2D_3D(make_napari_viewer, tmp_path, reference, moving):

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
            model=AffineTransformChoices.Euclidean,
            output=tmp_path / 'my_affine.txt'
            )

    viewer.layers['layer0_pts'].data = nuclei2D_3Dpts
    viewer.layers['layer1_pts'].data = nuclei3D_pts

    actual_affine = np.asarray(l1.affine)
    expected_affine = np.array([[1.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
           [0.000000e+00, 1.000000e+00, 5.780469e-17, 4.107270e-31],
           [0.000000e+00, -1.580578e-17, 1.000000e+00, 2.842171e-14],
           [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00]])

    """
    np.array([
        [ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  3.00000000e+01],
     [ 0.00000000e+00,  1.00000000e+00,  2.89023467e-17,  0.00000000e+00],
     [ 0.00000000e+00, -7.90288925e-18,  1.00000000e+00,  1.42108547e-14],
     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    """

    np.testing.assert_allclose(actual_affine, expected_affine, rtol=1e-06)









"""
layer0_pts = np.array([[140.38371886,
                        322.5390704], [181.91866481, 319.65803368],
                       [176.15659138, 259.1562627],
                       [140.14363246, 254.59462124]])
layer1_pts = np.array([[70.94741072,
                        117.37477536], [95.80911919, 152.00358359],
                       [143.16475439, 118.55866623],
                       [131.32584559, 83.33791256]])
                       
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

# 2D onto 2D tests
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
"""
"""
nuclei2D_3Dpts = np.array([[ 0.        ,  68.47649186,  67.08770344],
                       [ 0.        ,  85.14195298,  51.81103074],
                       [ 0.        , 104.58499096,  43.94122966],
                       [ 0.        , 154.58137432, 113.38065099]])
"""