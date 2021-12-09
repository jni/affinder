import argparse
import napari
from .affinder import start_affinder

parser = argparse.ArgumentParser()
parser.add_argument('filenames', help='Images to load', nargs='*')


def main():
    fns = parser.parse_args().filenames
    viewer = napari.Viewer()
    if len(fns) > 0:
        viewer.open(fns, stack=False)
    viewer.window.add_dock_widget(start_affinder(), area='right')
    napari.run()


if __name__ == '__main__':
    main()
