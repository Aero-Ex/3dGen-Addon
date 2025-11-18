"""
3D-Gen - Blender Addon
Generate 3D assets from images or text using Microsoft TRELLIS
"""

bl_info = {
    "name": "3D-Gen",
    "author": "AeroX",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > 3D-Gen",
    "description": "Generate 3D assets from images or text using TRELLIS AI",
    "category": "Add Mesh",
    "doc_url": "https://github.com/Aero-Ex",
}

import bpy
import sys
import os
from pathlib import Path
from bpy.props import (
    StringProperty,
    IntProperty,
    FloatProperty,
    BoolProperty,
    EnumProperty,
    PointerProperty,
)
from bpy.types import (
    Panel,
    Operator,
    PropertyGroup,
    AddonPreferences,
)

# Cache to avoid repeated path setup
_PATHS_INITIALIZED = False

def setup_venv_path():
    """Add virtual environment to sys.path if it exists (cached)"""
    global _PATHS_INITIALIZED
    if _PATHS_INITIALIZED:
        return True  # Already set up
        
    # Import from dependency_installer to use single source of truth
    from . import dependency_installer
    venv_path = dependency_installer.get_venv_path()

    if not venv_path.exists():
        return False

    # Determine site-packages location based on platform
    if os.name == 'nt':  # Windows
        site_packages = venv_path / 'Lib' / 'site-packages'
    else:  # Linux/Mac
        # Find the python3.x directory
        lib_path = venv_path / 'lib'
        if lib_path.exists():
            python_dirs = [d for d in lib_path.iterdir() if d.name.startswith('python3.')]
            if python_dirs:
                site_packages = python_dirs[0] / 'site-packages'
            else:
                return False
        else:
            return False

    # Add to sys.path FIRST (before Blender's packages)
    site_packages_str = str(site_packages)
    if site_packages.exists():
        # Remove if already in sys.path
        if site_packages_str in sys.path:
            sys.path.remove(site_packages_str)
        # Add at the VERY BEGINNING so venv packages are found first
        sys.path.insert(0, site_packages_str)
        print(f"3D-Gen: Added venv to sys.path: {site_packages_str}")
        _PATHS_INITIALIZED = True
        return True

    return False


def setup_bundled_trellis():
    """Add bundled TRELLIS package to sys.path (cached)"""
    global _PATHS_INITIALIZED
    if _PATHS_INITIALIZED:
        return True  # Already set up
        
    addon_dir = Path(__file__).parent
    addon_dir_str = str(addon_dir)

    # Add addon directory to sys.path so bundled trellis/ can be imported
    if addon_dir_str not in sys.path:
        sys.path.insert(0, addon_dir_str)
        print(f"3D-Gen: Added bundled package to sys.path: {addon_dir_str}")
        return True

    return False


# Set up venv path before importing modules (only runs once)
venv_added = setup_venv_path()

# Set up bundled TRELLIS package (only runs once)
setup_bundled_trellis()

# CRITICAL: Remove any modules loaded from Blender's Python (only runs once)
# This ensures we use the venv's packages, not Blender's
def cleanup_blender_modules():
    """Remove torch and numpy modules if they were loaded from Blender's Python"""
    global _PATHS_INITIALIZED
    if _PATHS_INITIALIZED:
        return  # Already cleaned up
        
    # Remove torch modules
    torch_modules = [key for key in sys.modules.keys() if key.startswith('torch')]
    if torch_modules:
        for module in torch_modules:
            del sys.modules[module]
        print(f"3D-Gen: Cleared {len(torch_modules)} torch modules from Blender's Python")
    
    # Remove numpy modules - CRITICAL for onnxruntime
    numpy_modules = [key for key in sys.modules.keys() if key.startswith('numpy')]
    if numpy_modules:
        for module in numpy_modules:
            del sys.modules[module]
        print(f"3D-Gen: Cleared {len(numpy_modules)} numpy modules from Blender's Python")
    
    _PATHS_INITIALIZED = True

if venv_added:
    cleanup_blender_modules()

    # Also, move Blender's site-packages to END of sys.path so venv takes priority
    blender_site_packages = []
    for i, path in enumerate(list(sys.path)):
        if 'blender' in path.lower() and 'site-packages' in path.lower():
            blender_site_packages.append((i, path))

    # Remove from current position and add to end
    for i, path in reversed(blender_site_packages):
        sys.path.remove(path)
        sys.path.append(path)

    if blender_site_packages:
        print(f"3D-Gen: Moved {len(blender_site_packages)} Blender site-packages to END of sys.path")

# Import addon modules
from . import operators
from . import ui
from . import preferences
from . import multi_image_ops

classes = (
    preferences.TRELLIS_AddonPreferences,
    operators.TRELLIS_ImageItem,
    operators.TRELLIS_Properties,
    operators.TRELLIS_OT_InstallDependencies,
    operators.TRELLIS_OT_InitializePipeline,
    operators.TRELLIS_OT_GenerateFromImage,
    operators.TRELLIS_OT_GenerateFromText,
    operators.TRELLIS_OT_GenerateVariant,
    operators.TRELLIS_OT_GenerateImageConsole,
    operators.TRELLIS_OT_GenerateTextConsole,
    operators.TRELLIS_OT_ImportLastGeneration,
    operators.TRELLIS_OT_OpenPreferences,
    multi_image_ops.TRELLIS_UL_ImageList,
    multi_image_ops.TRELLIS_OT_AddMultiImage,
    multi_image_ops.TRELLIS_OT_RemoveMultiImage,
    multi_image_ops.TRELLIS_OT_ClearMultiImages,
    multi_image_ops.TRELLIS_OT_GenerateFromMultiImage,
    multi_image_ops.TRELLIS_OT_GenerateMultiImageConsole,
    ui.TRELLIS_PT_MainPanel,
)

def auto_initialize():
    """Auto-initialize TRELLIS pipeline on startup if dependencies are installed"""
    print("TRELLIS: Auto-initialize function called")
    try:
        # Check if dependencies are installed
        from . import dependency_installer
        venv_path = dependency_installer.get_venv_path()
        
        print(f"TRELLIS: Checking venv path: {venv_path}")
        print(f"TRELLIS: Venv exists: {venv_path.exists()}")
        
        if not venv_path.exists():
            print("TRELLIS: Auto-initialize skipped - dependencies not installed")
            return None  # Return None to stop timer
        
        print("TRELLIS: Starting auto-initialization...")
        
        # Get preferences to check settings
        try:
            preferences = bpy.context.preferences.addons[__package__].preferences
            use_cuda = preferences.use_cuda
        except:
            # Fallback if context not available
            use_cuda = True
            print("TRELLIS: Using default CUDA setting (True)")
        
        # Initialize pipeline
        from .pipeline_manager import get_pipeline_manager
        manager = get_pipeline_manager()
        
        if not manager.initialized:
            print("TRELLIS: Initializing pipeline (this may take 1-2 minutes)...")
            success = manager.initialize(use_cuda=use_cuda)
            if success:
                print("TRELLIS: ✓ Auto-initialization complete! Ready to generate.")
                # Invalidate UI cache to show initialized status
                try:
                    from . import ui
                    ui.invalidate_status_cache()
                except:
                    pass
            else:
                print("TRELLIS: ⚠ Auto-initialization failed - click Initialize button manually")
        else:
            print("TRELLIS: Already initialized")
            
    except Exception as e:
        print(f"TRELLIS: Auto-initialize error: {e}")
        import traceback
        traceback.print_exc()
    
    return None  # Return None to stop the timer

def register():
    """Register addon classes and properties"""
    for cls in classes:
        bpy.utils.register_class(cls)

    # Add properties to Scene
    bpy.types.Scene.trellis_props = PointerProperty(type=operators.TRELLIS_Properties)
    
    # Removed auto-initialization on startup
    # Pipeline will initialize automatically when user clicks generate
    print("TRELLIS: Addon registered - pipeline will initialize on first generation")

def unregister():
    """Unregister addon classes and properties"""
    # Remove properties from Scene
    del bpy.types.Scene.trellis_props

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()
