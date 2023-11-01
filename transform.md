# Transformations
Lets assume some user has found a desired allignment and is ready to apply a transformation. 
This section discusses nuance when applying atransformation to an image.

## Scipy
When using `scipy.ndimage.affine_transform()` you must use the inverse transformation matrix i.e, the transformation from the reference image to the moving image. 

## Skimage
Similar to scipy, when using `skimage.transform.warp()`, you must use the inverse transfromation matrix.
Additionally, skimage uses X/Y coordinate conventions whereas napari uses NumPy like row/col coordinate conventions. This means that to use warp() correctly, you must first transpose the 0th and 1st row and column of the transformation matrix. e.g,

```python
def matrix_rc2xy(affine_matrix):
    swapped_cols = affine_matrix[:, [1, 0, 2]]
    swapped_rows = swapped_cols[[1, 0, 2], :]
    return swapped_rows
```

## Examples
### Scipy





Please note that when using skimage.transform.warp():

- napari uses row column coordinate conventions, whereas skimage.transform.warp uses XY coordinate conventions.
- Additionally, ndimage.affine_transform and skimage.transform.warp expect the inverse transformation matrix, that is, the transformation from the reference image to the moving image. This is because when you want to create a new image, you need to find a value for every target pixel, so you want to go from every new pixel coordinate to the place it came from in the image you're transforming.

please refer to the following code for reference on how to correctly use skimage.transform.warp():

```python
from skimage import data, transform
from scipy import ndimage as ndi
import napari
import numpy as np

image0 = data.camera()
image1 = transform.rotate(image0[100:, 32:496], 60)

viewer = napari.Viewer()
l0 = viewer.add_image(image0, colormap='bop blue', blending='additive')
l1 = viewer.add_image(image1, colormap='bop purple', blending='additive')

qtwidget, widget = viewer.window.add_plugin_dock_widget(
        'affinder', 'Start affinder'
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

def matrix_rc2xy(affine_matrix):
    swapped_cols = affine_matrix[:, [1, 0, 2]]
    swapped_rows = swapped_cols[[1, 0, 2], :]
    return swapped_rows

mat = np.asarray(l1.affine)
tfd_ndi = ndi.affine_transform(image1, np.linalg.inv(mat))
viewer.add_image(tfd_ndi, colormap='bop orange', blending='additive')
tfd_skim = transform.warp(image1, np.linalg.inv(matrix_rc2xy(mat)))
viewer.add_image(tfd_skim, colormap='bop orange', blending='additive', visible=False)

napari.run()
```
