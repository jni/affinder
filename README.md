# affinder

[![License](https://img.shields.io/pypi/l/affinder.svg?color=green)](https://github.com/napari/affinder/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/affinder.svg?color=green)](https://pypi.org/project/affinder)
[![Python Version](https://img.shields.io/pypi/pyversions/affinder.svg?color=green)](https://python.org)
[![tests](https://github.com/jni/affinder/workflows/tests/badge.svg)](https://github.com/jni/affinder/actions)
[![codecov](https://codecov.io/gh/jni/affinder/branch/master/graph/badge.svg)](https://codecov.io/gh/jni/affinder)

Quickly find the affine matrix mapping one image to another using manual correspondence points annotation

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/docs/plugins/index.html
-->

## Installation

You can install `affinder` via [pip]:

    pip install affinder


## How to use

Affinder is a napari plugin for quickly determining an affine transformation that can be used to register two images from a few manual point annotations.

To streamline the annotation process, it is divided into two phases:

1. In the first phase, The user annotates three points in the reference image using a new point layer in the viewer. The layer visibility then changes to the moving image and then annotates the corresponding points in the same order. (For volumes, the initial phase requires the annotation of four control point pairs.) As soon as enough control point pairs have been added to determine an initial affine transformation, this transformation
is applied in the napari viewer and the annotation mode changes.

2. In the second phase of the annotation, the visibility of the reference layer and the active point layer for annotation are toggled automatically after each additional point. When adding a point the cursor is automatically placed in the corrsponding position determined by the current affine transform. With every added control point pair the affine transform is dynamically updated.

Practical tips:

* For a good initial affine estimate, try and find control points close to three corners of an image. You want to avoid situations where the control points all lie close to a single line (degenerate case) or very close to each other (this does not constrain the possible transforms well.)
* Once the initial transform has been applied in the viewer, use the opactity slider of the image layers to blend both layers. This will quickly reveal problem areas in the images where the registration is not good. Try and improve the registration by adding control point pairs in these areas.





## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"affinder" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/jni/affinder/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
