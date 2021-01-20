import sys
import napari
from .affinder import start_affinder


def main():
    fns = sys.argv[1:]
    viewer = napari.Viewer()
    if len(fns) > 0:
        viewer.open(fns, stack=False)
    viewer.window.add_dock_widget(start_affinder, area='right')
    napari.run()


if __name__ == '__main__':
    main()
