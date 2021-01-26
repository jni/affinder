from napari_plugin_engine import napari_hook_implementation
from .affinder import start_affinder


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return start_affinder
