import numpy as np
from scipy import ndimage as ndi
from skimage import data, feature, filters, util, segmentation, morphology, transform
import toolz as tz

median_filter = tz.curry(ndi.median_filter)
remove_holes = tz.curry(morphology.remove_small_holes)
remove_objects = tz.curry(morphology.remove_small_objects)


@tz.curry
def threshold_with(image, method=filters.threshold_li):
    return image > method(image)


to_origin = np.array([0, -127.5, -127.5])
c = np.cos(np.radians(60))
s = np.sin(np.radians(60))
rot60 = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
        ])
from_origin = -to_origin
trans = np.array([0, 5, 10])

nuclei = data.cells3d()[:, 1, ...]
nuclei_rotated = ndi.rotate(nuclei, 60, axes=(1, 2), reshape=False)
nuclei_rotated_translated = ndi.shift(nuclei_rotated, trans)
nuclei_points = feature.peak_local_max(filters.gaussian(nuclei, 15))

nuclei_points_rotated_translated = ((nuclei_points+to_origin) @ rot60.T
                                    + from_origin + trans)

nuclei_binary = tz.pipe(
        nuclei,
        median_filter(size=3),
        threshold_with(method=filters.threshold_li),
        remove_holes(area_threshold=20**3),
        remove_objects(min_size=20**3),
        )
nuclei_labels = segmentation.watershed(
        filters.farid(nuclei),
        markers=util.label_points(nuclei_points, nuclei.shape),
        mask=nuclei_binary,
        )
nuclei_labels_rotated = ndi.rotate(
        nuclei_labels, 60, axes=(1, 2), reshape=False, order=0
        )
nuclei_labels_rotated_translated = ndi.shift(nuclei_labels, trans, order=0)

nuclei2d = nuclei[30]
nuclei2d_points = nuclei_points[:, 1:]  # remove z = project onto yx
nuclei2d_rotated = nuclei_rotated[30]
nuclei2d_rotated_translated = nuclei_rotated_translated[30]
nuclei2d_labels = nuclei_labels[30]
nuclei2d_labels_rotated_translated = nuclei_labels_rotated_translated[30]
nuclei2d_points_rotated_translated = nuclei_points_rotated_translated[:, 1:]

if __name__ == '__main__':
    import napari
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(nuclei, blending='additive')
    viewer.add_points(nuclei_points)
    viewer.add_labels(nuclei_labels, blending='translucent_no_depth')
    viewer.add_image(nuclei_rotated_translated, blending='additive')
    viewer.add_points(nuclei_points_rotated_translated, face_color='red')
    viewer.add_labels(nuclei_labels_rotated, blending='translucent_no_depth')

    viewer.grid.enabled = True
    viewer.grid.stride = 3
    napari.run()
