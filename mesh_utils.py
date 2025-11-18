"""
Utilities for importing TRELLIS meshes into Blender
"""

import bpy
import bmesh
import os
import tempfile
from typing import Optional


def import_glb(filepath: str, collection_name: str = "TRELLIS") -> Optional[bpy.types.Object]:
    """
    Import GLB file into Blender

    Args:
        filepath: Path to GLB file
        collection_name: Name of collection to add object to

    Returns:
        Imported object or None on error
    """
    try:
        # Get or create collection
        if collection_name not in bpy.data.collections:
            collection = bpy.data.collections.new(collection_name)
            bpy.context.scene.collection.children.link(collection)
        else:
            collection = bpy.data.collections[collection_name]

        # Store objects before import
        objects_before = set(bpy.data.objects)

        # Import GLB
        bpy.ops.import_scene.gltf(filepath=filepath)

        # Find newly imported objects
        objects_after = set(bpy.data.objects)
        new_objects = objects_after - objects_before

        if not new_objects:
            print("Warning: No objects imported")
            return None

        # Move objects to collection and get main object
        main_obj = None
        for obj in new_objects:
            # Remove from all collections
            for coll in obj.users_collection:
                coll.objects.unlink(obj)

            # Add to our collection
            collection.objects.link(obj)

            # Select mesh as main object
            if obj.type == 'MESH' and main_obj is None:
                main_obj = obj

        # Select and activate main object
        if main_obj:
            bpy.ops.object.select_all(action='DESELECT')
            main_obj.select_set(True)
            bpy.context.view_layer.objects.active = main_obj
            
            # Ensure object is visible in all viewports
            main_obj.hide_viewport = False
            main_obj.hide_render = False
            main_obj.hide_set(False)
            
            # Frame the object in view (zoom to fit)
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    for region in area.regions:
                        if region.type == 'WINDOW':
                            override = {'area': area, 'region': region}
                            with bpy.context.temp_override(**override):
                                bpy.ops.view3d.view_selected()
                            break

        return main_obj

    except Exception as e:
        print(f"Error importing GLB: {e}")
        import traceback
        traceback.print_exc()
        return None


def import_mesh_from_trellis(
    outputs: dict,
    name: str = "TRELLIS_Object",
    temp_dir: Optional[str] = None,
    texture_size: int = 1024,
    simplify: float = 0.95,
) -> Optional[bpy.types.Object]:
    """
    Import mesh from TRELLIS outputs

    Args:
        outputs: TRELLIS pipeline outputs
        name: Name for the imported object
        temp_dir: Temporary directory for GLB export
        texture_size: Texture resolution for mesh export
        simplify: Mesh simplification ratio (0-1)

    Returns:
        Imported Blender object or None on error
    """
    try:
        # Use system temp dir if not specified
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()

        # Create temporary GLB file
        glb_path = os.path.join(temp_dir, f"{name}.glb")

        # Export from TRELLIS to GLB
        from .pipeline_manager import get_pipeline_manager
        manager = get_pipeline_manager()

        success = manager.export_to_glb(
            outputs,
            glb_path,
            simplify=simplify,
            texture_size=texture_size,
        )

        if not success:
            return None

        # Import GLB into Blender
        obj = import_glb(glb_path, collection_name="TRELLIS")

        if obj:
            obj.name = name

        # Clean up temp file
        try:
            os.remove(glb_path)
        except:
            pass

        return obj

    except Exception as e:
        print(f"Error importing mesh from TRELLIS: {e}")
        import traceback
        traceback.print_exc()
        return None


def center_object(obj: bpy.types.Object):
    """
    Center object at world origin

    Args:
        obj: Blender object to center
    """
    try:
        # Set origin to geometry
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

        # Move to world origin
        obj.location = (0, 0, 0)

    except Exception as e:
        print(f"Error centering object: {e}")


def scale_object_to_size(obj: bpy.types.Object, target_size: float = 2.0):
    """
    Scale object to fit within a target size

    Args:
        obj: Blender object to scale
        target_size: Target size (max dimension)
    """
    try:
        # Get current size
        max_dim = max(obj.dimensions)

        if max_dim > 0:
            # Calculate scale factor
            scale_factor = target_size / max_dim

            # Apply scale
            obj.scale = (scale_factor, scale_factor, scale_factor)

    except Exception as e:
        print(f"Error scaling object: {e}")
