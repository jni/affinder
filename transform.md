Lets say we have used affinder to regester out moving image with the reference image and now we wish to use the resulting transform matrix for downstream analysis.
The guide describes the required changes to make the matrix compatable with scipy and skimage transformations.

## Getting an affinder matrix
Consider some image and its respective affine matrix. for this example we will use skimag's data.camera.
The code below loads the image and then rotates it to generate our moving image. 
We then programatically lauch th affinder widget and add some points to get our transform matrix

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

mat = np.asarray(l1.affine)
```

At this point, `mat` holds the information required to transform the **moving** image to the **reference** image.
Before this matrix can be used however, it requires some processing which differs depending on which transformation method is to be used.

## Scipy

`scipy.ndimage.affine_transform()` expects a matrix that transforms from the **reference** image to the **moving** image, therefore we should use the inverse matrix from our affinder output. That's because when you want to create a new image, you need to find a value for every target pixel, so you want to go from every new pixel coordinate to the place it came from in the image you're transforming.

```python
tfd_ndi = ndi.affine_transform(image1, np.linalg.inv(mat))
viewer.add_image(tfd_ndi, colormap='bop orange', blending='additive')
```

## Skimage
Similar to scipy, when using `skimage.transform.warp()`, it transforms from reference to moving, so you must use the inverse transfromation matrix.
Additionally, skimage uses X/Y coordinate conventions whereas napari uses NumPy-like row/col coordinate conventions. This means that to use warp() correctly, you must first transpose the 0th and 1st row and column of the transformation matrix, as below.

```python
def matrix_rc2xy(affine_matrix):
    swapped_cols = affine_matrix[:, [1, 0, 2]]
    swapped_rows = swapped_cols[[1, 0, 2], :]
    return swapped_rows

tfd_skim = transform.warp(image1, np.linalg.inv(matrix_rc2xy(mat)))
viewer.add_image(tfd_skim, colormap='bop orange', blending='additive', visible=False)
```

This should be fixed in skimage 2.0 as it moves fully to NumPy-like coordinates.
