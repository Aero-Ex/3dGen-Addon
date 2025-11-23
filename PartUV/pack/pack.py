try:    
    from uv_distortion import *
except:
    from .uv_distortion import *
import numpy as np
import sys
import subprocess
import matplotlib.pyplot as plt
import os
import argparse
from scipy.spatial import cKDTree
import time

from .eval_charts import evaluate_mesh


def generate_unique_colors(num_ids):
    cmap = plt.get_cmap('tab20')  # You can choose other colormaps like 'hsv', 'jet', etc.
    colors = [cmap(i % 20)[:3] for i in range(num_ids)]  # RGB colors
    return colors

def normalize_uv_charts(uv_charts):
    all_uvs = np.vstack(uv_charts)

    min_x, min_y = all_uvs.min(axis=0)
    max_x, max_y = all_uvs.max(axis=0)

    range_x = max_x - min_x
    range_y = max_y - min_y

    # Avoid division by zero if everything is collapsed on an axis
    if range_x == 0 or range_y == 0:
        print("Warning: Zero range in one dimension; UVs cannot be rescaled meaningfully.")
        return uv_charts

    scale = 1.0 / max(range_x, range_y)
    for i, chart in enumerate(uv_charts):
        uv_charts[i] = (chart - [min_x, min_y]) * scale
    return uv_charts


def calculate_uv_utilization(mesh):
        
    uv = mesh.visual.uv

    # Extract face indices and UV coordinates
    faces = mesh.faces
    uv_coords = uv
    face_uv = uv_coords[faces]

    # Compute UV areas using the shoelace formula
    def compute_uv_areas(face_uv):
        a = face_uv[:, 0, :]
        b = face_uv[:, 1, :]
        c = face_uv[:, 2, :]
        areas = 0.5 * np.abs(
            a[:, 0]*(b[:, 1] - c[:, 1]) +
            b[:, 0]*(c[:, 1] - a[:, 1]) +
            c[:, 0]*(a[:, 1] - b[:, 1])
        )
        return areas

    uv_areas = compute_uv_areas(face_uv)
    total_uv_area = np.sum(uv_areas)
    print(f"Total UV area: {total_uv_area}")


def color_mesh_components(components, color_list=None):
    """
    Colors each component of a mesh with a different color.
    
    Parameters:
    -----------
    mesh : trimesh.Trimesh
        The input mesh to be colored
    color_list : list of RGBA colors, optional
        List of colors to use. If None, random colors will be generated.
        Colors should be in RGBA format with values between 0 and 1.
    
    Returns:
    --------
    trimesh.Trimesh
        A new mesh with colored components
    """

    if type(components) is trimesh.Trimesh:
        components = components.split(only_watertight=False)
    
    # If no colors provided, generate random colors
    if color_list is None:
        # Generate random colors (excluding alpha)
        color_list = np.random.random((len(components), 4))
        # Set alpha to 1.0 for all colors
        color_list[:, 3] = 1.0
    
    # Make sure we have enough colors
    if len(color_list) < len(components):
        raise ValueError(f"Not enough colors provided. Need {len(components)} colors but got {len(color_list)}")
    
    # Create a list to store colored meshes
    colored_components = []
    face_colors = []
    
    # Color each component
    for idx, component in enumerate(components):
        # Create a color array for all faces in this component
        component_color = np.tile(list(color_list[idx]) + [1], (len(component.faces), 1))
        component.visual.face_colors = component_color
        colored_components.append(component)
        face_colors.append(component_color)
    
    # Combine all components back into a single mesh
    colored_mesh = trimesh.util.concatenate(colored_components)
    
    # Set the combined face colors
    colored_mesh.visual.face_colors = np.vstack(face_colors)
    return colored_mesh



import hashlib
def get_mesh_hash(tri_mesh):
    """
    Compute an MD5 hash of the mesh vertices and faces.
    This helps create a (mostly) unique signature for each mesh part.
    """
    # Convert vertices (float64) and faces (int64) to bytes
    vert_bytes = tri_mesh.vertices.view(np.float64).tobytes()
    # face_bytes = tri_mesh.faces.view(np.int64).tobytes()

    # Combine both for hashing
    # combined = vert_bytes + face_bytes
    
    # Generate the MD5 hash
    return hashlib.md5(vert_bytes).hexdigest()

def pack_mesh(mesh_path, uvpackmaster=False, save_visuals=False, eval_mesh=False):
    time1 = time.time()
    time_dict = {}

    if uvpackmaster:
        # print(f"Running blenderproc to pack mesh: {mesh_path}")
        current_file = os.path.abspath(__file__)
        uvpackmaster_script = os.path.join(os.path.dirname(current_file), "pack_parts.py")
        subprocess.run(
            ["blenderproc", "run", uvpackmaster_script, "--input_dir", os.path.join(mesh_path, "individual_parts")],
            stdout=sys.stdout,
            stderr=subprocess.STDOUT,
            check=True
        )
    else:
        from pack.pack_blender import pack_blender
        pack_blender(mesh_path)
        
    time_pack_ms = (time.time() - time1) * 1000.0
    time_dict["pack_ms"] = time_pack_ms
    if save_visuals:
        time_save_visuals_start = time.time()
        all_parts = [trimesh.load(os.path.join(mesh_path, f)) for f in os.listdir(mesh_path) if f.endswith("_packed.obj")]
        colors = generate_unique_colors(len(all_parts))

        hash_color = {}

        for color, mesh_file in zip(colors, all_parts):
            # Split the loaded file into connected components
            components = mesh_file.split(only_watertight=False)
            
            # For each connected component, compute a hash and store color
            for comp in components:
                h = get_mesh_hash(comp)
                hash_color[h] = color


        uv_mesh = trimesh.load(os.path.join(mesh_path, "final_packed.obj"))
        
            
            
        uv_mesh.visual.uv = normalize_uv_charts([uv_mesh.visual.uv])[0]
        # uv_mesh.export(os.path.join(mesh_path, "final_packed.obj"))

        uv_mesh_components = uv_mesh.split(only_watertight=False)

        all_uv = []
        new_uv_color = []
        for mesh_part in uv_mesh_components:
            # Compute the same hash
            part_hash = get_mesh_hash(mesh_part)
            # Look up the corresponding color
            part_color = hash_color.get(part_hash, (0, 0, 0))
            
            # Store UV and color
            all_uv.append(mesh_part.visual.uv)
            new_uv_color.append(part_color)

        
        method = "uvpackmaster" if uvpackmaster else "blender"
        
        save_uv_layout_with_packing(uv_mesh_components, all_uv, os.path.join(mesh_path, f"final_packed_uv_{method}.png"), mode=new_uv_color,
                                    image_size=8192, save_image=True)
        save_uv_layout_with_packing(uv_mesh_components, all_uv, os.path.join(mesh_path, f"distortion_{method}.png"), mode='area',
                                image_size=8192, save_image=True)
        
        print(f"saved uv layout to {os.path.join(mesh_path, f'final_packed_uv_{method}.png')}")

        # Only save the mesh_color.ply if uvpackmaster is used and we have part information
        if uvpackmaster:
            mesh_colored = color_mesh_components(uv_mesh_components, new_uv_color)
            mesh_colored = trimesh.Trimesh(vertices=mesh_colored.vertices, faces=mesh_colored.faces, 
                                            face_colors=mesh_colored.visual.face_colors)
            calculate_uv_utilization(uv_mesh)
            mesh_colored.export(os.path.join(mesh_path, "mesh_color.ply"))
        time_save_visuals_end = time.time()
        time_save_visuals_ms = (time_save_visuals_end - time_save_visuals_start) * 1000.0
        time_dict["save_visuals_ms"] = time_save_visuals_ms
        
    if eval_mesh:
        metrics = evaluate_mesh(os.path.join(mesh_path, "final_packed.obj"))
        print(metrics)

    return time_dict
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partuv_output_path', '-p', type=str, required=True, help='Path to partuv output directory')
    parser.add_argument("--uvpackmaster", '-U', action="store_true")
    parser.add_argument("--save_visuals", '-S', action="store_true")
    parser.add_argument("--eval_mesh", '-E', action="store_true", default=False, help="Evaluate the mesh after packing")
    args = parser.parse_args()

    pack_mesh(args.partuv_output_path, args.uvpackmaster, args.save_visuals, args.eval_mesh)

if __name__ == "__main__":
    main()
