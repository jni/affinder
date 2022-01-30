from affinder import start_affinder
from affinder.affinder import AffineTransformChoices
from skimage import data, transform
import numpy as np


def test_basic(make_napari_viewer, tmp_path):
    image0 = data.camera()
    image1 = transform.rotate(image0[100:, 32:496], 60)

    viewer = make_napari_viewer()

    l0 = viewer.add_image(image0, colormap='green', blending='additive')
    l1 = viewer.add_image(image1, colormap='magenta', blending='additive')

    my_widget_factory = start_affinder()
    my_widget_factory(
            viewer=viewer,
            reference=l0,
            moving=l1,
            model=AffineTransformChoices.affine,
            output=tmp_path / 'my_affine.txt'
            )

    viewer.layers['image0_pts'].data = np.array([[148.19396647, 234.87779732],
                                                 [484.56804381, 240.55720892],
                                                 [474.77521025, 385.88403205]])
    viewer.layers['image1_pts'].data = np.array([[150.02534429, 80.65355322],
                                                 [314.75696913, 375.13825634],
                                                 [184.33085012, 439.81718637]])
    actual = np.asarray(l1.affine)
    expected = np.array(  # yapf: ignore
            [[0.50221294, 0.86131375, 3.38128256],
             [-0.86478707, 0.50303866, 324.04591946],
             [0., 0., 1.]])

    np.testing.assert_allclose(actual, expected)

def test_image_shape(make_napari_viewer, tmp_path):
    image0 = data.camera()
    viewer = make_napari_viewer()

    l0 = viewer.add_image(image0, colormap='green', blending='additive')
    l1 = viewer.add_shapes([np.array([[0,0], [0,10], [10,10], [10,0]])],
                           face_color='magenta', blending='additive',
                           name="image1")

    my_widget_factory = start_affinder()
    my_widget_factory(
            viewer=viewer,
            reference=l0,
            moving=l1,
            model=AffineTransformChoices.affine,
            output=tmp_path / 'my_affine.txt'
            )

    viewer.layers['image0_pts'].data = np.array([[139.65538415, 256.33044259],
                                                [139.65538415, 329.40805331],
                                                [182.00718127, 321.10377937],
                                                [189.48102782, 260.48257956]])
    viewer.layers['image1_pts'].data = np.array([[0,0],
                                                 [0,10],
                                                 [10,10],
                                                 [10,0]])
    actual = np.asarray(l1.affine)
    expected = np.array(
        [[4.63336823e+00, -3.75678505e-01, 1.41411296e+02],
         [-2.08710281e-01, 6.72047104e+00, 2.59272410e+02],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    np.testing.assert_allclose(actual, expected)


def test_shape_shape(make_napari_viewer, tmp_path):
    viewer = make_napari_viewer()

    l0 = viewer.add_shapes([np.array([[139.65538415, 256.33044259],
                                      [139.65538415, 329.40805331],
                                      [182.00718127, 321.10377937],
                                      [189.48102782, 260.48257956]])],
                           face_color='green', blending='additive',
                           name="image0")
    l1 = viewer.add_shapes([np.array([[0,0], [0,10], [10,10], [10,0]])],
                           face_color='magenta', blending='additive',
                           name="image1")

    my_widget_factory = start_affinder()
    my_widget_factory(
            viewer=viewer,
            reference=l0,
            moving=l1,
            model=AffineTransformChoices.affine,
            output=tmp_path / 'my_affine.txt'
            )

    viewer.layers['image0_pts'].data = np.array([[139.65538415, 256.33044259],
                                                [139.65538415, 329.40805331],
                                                [182.00718127, 321.10377937],
                                                [189.48102782, 260.48257956]])
    viewer.layers['image1_pts'].data = np.array([[0,0],
                                                 [0,10],
                                                 [10,10],
                                                 [10,0]])
    actual = np.asarray(l1.affine)
    expected = np.array(
        [[4.63336823e+00, -3.75678505e-01, 1.41411296e+02],
         [-2.08710281e-01, 6.72047104e+00, 2.59272410e+02],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    np.testing.assert_allclose(actual, expected)