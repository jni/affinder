import affinder.plugin
from skimage import data, transform
import numpy as np


def test_basic(make_napari_viewer, napari_plugin_manager):
    image0 = data.camera()
    image1 = transform.rotate(image0[100:, 32:496], 60)

    napari_plugin_manager.register(affinder.plugin, name='affinder')
    viewer = make_napari_viewer()

    l0 = viewer.add_image(image0, colormap='green', blending='additive')
    l1 = viewer.add_image(image1, colormap='magenta', blending='additive')

    qtwidget, widget = viewer.window.add_plugin_dock_widget(
            'affinder', 'start_affinder'
            )
    widget.reference.bind(l0)
    widget.moving.bind(l1)
    widget()

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
