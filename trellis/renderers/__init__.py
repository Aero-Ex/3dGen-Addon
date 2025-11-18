import importlib

__attributes = {
    'OctreeRenderer': 'octree_renderer',
    'GaussianRenderer': 'gaussian_render',
    'MeshRenderer': 'mesh_renderer',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            try:
                module = importlib.import_module(f".{module_name}", __name__)
            except (ImportError, ValueError):
                module = importlib.import_module(f"trellis.renderers.{module_name}")
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            try:
                module = importlib.import_module(f".{name}", __name__)
            except (ImportError, ValueError):
                module = importlib.import_module(f"trellis.renderers.{name}")
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
    from .octree_renderer import OctreeRenderer
    from .gaussian_render import GaussianRenderer
    from .mesh_renderer import MeshRenderer