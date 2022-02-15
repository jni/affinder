from skimage import data, transform
import napari
import numpy as np

image0 = data.camera()
image1 = transform.rotate(image0[100:, 32:496], 60)

viewer = napari.Viewer()
l0 = viewer.add_image(image0, colormap='green', blending='additive')
l1 = viewer.add_image(image1, colormap='magenta', blending='additive')
l2 = viewer.add_points([[0, 0]])

qtwidget, widget = viewer.window.add_plugin_dock_widget(
        'affinder', 'Start affinder'
        )

napari.run()
