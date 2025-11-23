import blenderproc as bproc
import os
import trimesh
import copy
import numpy as np
import sys
import subprocess   
import os
import json
import argparse

# We will need to access bpy for the low-level UV packing operators
import bpy
import bmesh
import re
import math
from mathutils import Vector
# from tqdm import tqdm

# 1) Initialize BlenderProc in headless (no display) mode
bproc.init()

def import_obj(filepath, new_name="my_object"):
    # Load the OBJ file
    bpy.ops.wm.obj_import(filepath=filepath)
    
    # The newly imported object will be selected automatically,
    # so we can retrieve it with:
    obj = bpy.context.selected_objects[0]
    
    # Now rename the object
    obj.name = new_name
    return obj




def polygon_area_2d(coords):
    """
    Compute the 2D polygon area using the Shoelace formula.
    coords should be a list of (x, y) tuples.
    """
    area = 0.0
    n = len(coords)
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1] - coords[j][0] * coords[i][1]
    return abs(area) / 2.0

def get_uv_face_center(uv_coords):
    """
    Compute the centroid of a polygon in UV space.
    """
    x_sum = sum(coord[0] for coord in uv_coords)
    y_sum = sum(coord[1] for coord in uv_coords)
    n = len(uv_coords)
    return (x_sum / n, y_sum / n)

def scale_uvs_to_match_3d_area(obj):
    """
    Scale each face's UVs so its 2D (UV) area matches its 3D face area.
    This will scale each face independently, which may cause seams
    on connected edges of adjacent faces.
    """
    me = obj.data
    # bm = bmesh.new()
    # bm.from_mesh(me)
    bm = bmesh.from_edit_mesh(me)
    bm.faces.ensure_lookup_table()
    
    # Get the active UV layer
    uv_layer = bm.loops.layers.uv.active
    if not uv_layer:
        print("No active UV layer found. Aborting.")
        bm.free()
        return
    
    for face in bm.faces:
        # -- 1) Calculate the 3D face area
        area_3d = face.calc_area()
        if area_3d == 0:
            continue
        
        # -- 2) Gather the face’s UVs and calculate the current 2D area
        uv_coords = [loop[uv_layer].uv[:] for loop in face.loops]
        area_2d = polygon_area_2d(uv_coords)
        if area_2d == 0:
            continue
        
        # -- 3) Compute scale factor = sqrt(area_3d / area_2d)
        scale_factor = math.sqrt(area_3d / area_2d)
        
        # -- 4) Get face centroid in UV space, and scale about that point
        center_uv = get_uv_face_center(uv_coords)
        
        for loop in face.loops:
            # Original UV
            uv = loop[uv_layer].uv
            # Translate so center is at origin
            uv -= Vector(center_uv)
            # Scale
            uv *= scale_factor
            # Translate back
            uv += Vector(center_uv)
    
    # Write changes back to mesh
    # bm.to_mesh(me)
    # bm.free()
    
    # # Refresh the UVs in the viewport
    # me.update()
    bmesh.update_edit_mesh(me)





parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)

args = parser.parse_args()

directory = args.input_dir  

# Get a sorted list of all OBJ parts in the directory
# part_files = [f for f in os.listdir(directory) if re.match(r'^part_\d+\.obj$', f)]

# part_files = [os.path.join(f, "mesh.obj") for f in os.listdir(directory) if f.startswith("part_") and os.path.isdir(os.path.join(directory, f))]
part_files = [f for f in os.listdir(directory) if f.startswith("part_") and f.endswith(".obj") and not f.endswith("_packed.obj")]
part_files.sort()

print(f"Packing {len(part_files)} parts in {directory}")

# Clean up the scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

imported_objects = []

# Import and process each part
uv_offset = 0.0
uv_offset_step = 1.2  # how much to move each part in UV space to avoid overlap

for i, part_file in enumerate(part_files):
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    filepath = os.path.join(directory, part_file)
    
    # Assume the imported part is the active object
    obj = import_obj(filepath)
    
    bpy.context.view_layer.objects.active = obj

    # Go to EDIT mode to work with UVs
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # Select all UVs and pack islands
    bpy.ops.uv.select_all(action='SELECT')

    bpy.ops.uv.average_islands_scale()

    bpy.ops.uv.pack_islands(margin=0.001, scale=False)
    # save the obj with packed uv
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.wm.obj_export(filepath=os.path.join(os.getcwd(), f"{filepath[:-4]}_packed.obj"),
                          export_materials=False,export_normals=False)

print("All parts have been packed.")

# breakpoint()
# load all mesh in folder
# for each mesh, pack the UV


mesh_folder = directory

json_output_path = os.path.join(mesh_folder, 'faces.json')
if os.path.exists(json_output_path):
    os.remove(json_output_path)
    
    
# part_files = [os.path.join(f, "mesh_packed.obj") for f in os.listdir(directory) if f.startswith("part_") and os.path.isdir(os.path.join(directory, f))]
part_files = [f for f in os.listdir(directory) if f.startswith("part_") and   f.endswith("_packed.obj")]

part_files.sort()

for mesh_path in part_files:
    # if mesh_path.endswith('_packed.obj'):
    # if "mesh_packed.obj" in mesh_path:
        
    full_mesh_path = os.path.join(directory, mesh_path)
    # Load and process each mesh file
    
    # if '30' in mesh_path:
    #     breakpoint()
    mesh = trimesh.load_mesh(full_mesh_path, process=False)
    components = mesh.split(only_watertight=False,repair=False)
    
    if sum([c.vertices.shape[0] for c in components]) != mesh.vertices.shape[0]:
        mesh_merged = mesh.copy()
        print(f"SPLIT FAILED: {mesh_path}")
        components = [mesh_merged]
    else:
        mesh_merged = trimesh.util.concatenate(components)
    # Further processing as needed

    # 2. Combine ALL components into one mesh (the "original" shape)
    # mesh_merged = trimesh.util.concatenate(components)

    # 3. We need to know where (in the merged mesh) each component's vertices end up.
    #    We'll build a list of per-component vertex offsets:
    vertex_offsets = []
    face_offsets = []
    cumulative_vertex_offset = 0
    cumulative_face_offset = 0
    for c in components:
        vertex_offsets.append(cumulative_vertex_offset)
        face_offsets.append(cumulative_face_offset)
        cumulative_vertex_offset += len(c.vertices)
        cumulative_face_offset += len(c.faces)
        
    face_list = []
    
    first_face = mesh.faces[face_offsets[0]]
    num_new_face = 0
    for face_num in face_offsets[1:]:
        # if face_num == 88:
        #     breakpoint()
        
        # breakpoint()
        face_list.append([first_face[0], first_face[1], mesh_merged.faces[face_num][0]])
        face_list.append([first_face[1], mesh_merged.faces[face_num][0], mesh_merged.faces[face_num][1]])
        num_new_face += 2
        
        
    
    new_face = np.array(face_list)

    if len(new_face) != 0:
        faces_with_new = np.vstack([mesh_merged.faces, new_face])
    else:
        faces_with_new = mesh_merged.faces
        
    # Build a new Trimesh object with the additional face
    mesh_with_new_face = trimesh.Trimesh(
        vertices=mesh_merged.vertices,
        faces=faces_with_new,
        process=False
    )
    
    
    mesh_with_new_face.visual = copy.deepcopy(mesh_merged.visual)
    
    # Normalize UVs with scale (3D area / 2D area)
    total_3d_area = mesh_merged.area
    # breakpoint()
    total_2d_area = sum(polygon_area_2d(mesh_with_new_face.visual.uv[face]) for face in mesh_with_new_face.faces)
    scale_factor = total_3d_area / total_2d_area
    mesh_with_new_face.visual.uv *= math.sqrt(scale_factor)
    
    mesh_with_new_face.export(full_mesh_path)
    
    # Save the number of new faces to a JSON file
    if os.path.exists(json_output_path):
        with open(json_output_path, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = {}

    existing_data[mesh_path[:-4]] = {
        'num_face': len(mesh.faces),
        'num_new_face': num_new_face
    }
    with open(json_output_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)


# breakpoint()

def delete_last_n_faces(obj, n):
    """
    Deletes the last n faces from the given object's mesh.
    Assumes object is a mesh object.
    """
    # Ensure the object is active and in Edit Mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    mesh = obj.data
    bm = bmesh.from_edit_mesh(mesh)
    
    total_faces = len(bm.faces)
    if total_faces < n:
        print(f"Object '{obj.name}' only has {total_faces} faces, fewer than {n}. Deleting them all.")
        n = total_faces

    # Convert bm.faces to a list so we can slice
    faces_to_delete = list(bm.faces)[-n:]
    
    # Select faces to delete
    for f in faces_to_delete:
        f.select = True
    
    # Delete faces
    bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES')
    
    # Write changes back to mesh
    bmesh.update_edit_mesh(mesh)
    
    # Return to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

def remove_faces_from_json(json_filepath):
    """
    Reads the specified JSON file. The JSON is expected to contain a dict
    { "object_name": number_of_faces_to_remove, ... }
    
    For each entry:
      - Finds the object in the scene
      - Removes the specified number of faces from the end of its mesh
    """
    if not os.path.exists(json_filepath):
        print(f"JSON file not found at path: {json_filepath}")
        return
    with open(json_filepath, "r") as f:
        data = json.load(f)
    
    for object_name, face_data in data.items():
        # Attempt to get the object by name
        obj = bpy.data.objects.get(object_name)
        if not obj:
            print(f"Object '{object_name}' not found in the scene. Skipping...")
            continue
        
        # Check if it's a mesh object
        if obj.type != 'MESH':
            print(f"Object '{object_name}' is not a MESH. Skipping...")
            continue
        
        num_new_face = face_data.get("num_new_face", 0)
        if num_new_face > 0:
            print(f"Removing {num_new_face} faces from {object_name}...")
            delete_last_n_faces(obj, num_new_face)
            
        # Save the object with packed UVs
        save_path = os.path.join(directory, f"{object_name}.obj")
        bpy.ops.object.mode_set(mode='OBJECT')
        
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.wm.obj_export(filepath=save_path,  export_selected_objects=True, export_materials=False, export_normals=False)
        print(f"Saved {object_name} with packed UVs to {save_path}")
        bpy.ops.object.mode_set(mode='EDIT')


# Get a sorted list of all OBJ parts in the directory
# part_files = [f for f in os.listdir(directory) if f.startswith("part_")]

# breakpoint()
# Clean up the scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

imported_objects = []

# Import and process each part
uv_offset = 0.0
uv_offset_step = 1.2  # how much to move each part in UV space to avoid overlap

for i, part_file in enumerate(part_files):
    if part_file.endswith('_packed.obj'):
        filepath = os.path.join(directory, part_file)
        
        # Assume the imported part is the active object
        obj = import_obj(filepath, part_file[:-4])
    # bpy.context.view_layer.objects.active = obj

    # Go to EDIT mode to work with UVs
bpy.ops.object.select_all(action='SELECT')
    
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')

# Select all UVs and pack islands
bpy.ops.uv.select_all(action='SELECT')
# bpy.ops.uv.average_islands_scale()

bpy.ops.uv.pack_islands(margin=0.0025)
# save the obj with packed uv



# Provide your own path to the JSON file
json_file_path = os.path.join(directory, "faces.json")

print(json_file_path,json_output_path)
# Call the function
remove_faces_from_json(json_file_path)


bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='SELECT')

bpy.ops.wm.obj_export(filepath=os.path.join(os.path.dirname(directory), f"final_packed.obj"),
                        export_materials=False,export_normals=False)
                        # Save each imported object individually


print("All UVs have been final-packed.")
