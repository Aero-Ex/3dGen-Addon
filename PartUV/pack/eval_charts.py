#!/usr/bin/env python3

import sys
from collections import defaultdict
try:
    from uv_distortion import (
        save_uv_layout_no_normal,
        calculate_distortion_area,
        calculate_uv_face_area,
        calculate_distortion_area_nuvo,
        conformal_metric,
        save_uv_layout_no_normal_NUVO,
        stretch_metrics,
    )
except:
    from .uv_distortion import (
        save_uv_layout_no_normal,
        calculate_distortion_area,
        calculate_uv_face_area,
        calculate_3d_face_area,
        calculate_distortion_area_nuvo,
        conformal_metric,
        save_uv_layout_no_normal_NUVO,
        stretch_metrics,
    )
import os
import sys
import trimesh
from collections import defaultdict
import os, re, shutil, sys
import math

# import overlap
import torch
import glob
import numpy as np
import json
from tqdm import tqdm


class UnionFind:
    """
    A simple Disjoint Set (Union-Find) structure
    to group faces into connected components.
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            # Union by rank
            if self.rank[rx] < self.rank[ry]:
                self.parent[rx] = ry
            elif self.rank[rx] > self.rank[ry]:
                self.parent[ry] = rx
            else:
                self.parent[ry] = rx
                self.rank[rx] += 1




def uv_seam_length(mesh: trimesh.Trimesh) -> float:
    """
    Seam length (sum of UV-chart boundary lengths) – fully vectorized.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Must contain per-vertex UVs (`mesh.visual.uv`).

    Returns
    -------
    float
        Seam length in UV parameter space.
    """
    uv = mesh.visual.uv
    if uv is None:
        raise ValueError("Mesh contains no UV coordinates")
    uv = uv.astype(np.float64)                     # (V, 2)

    F = mesh.faces                                 # (F, 3)

    # ---- 1. build the 3*F edge list -----------------------------------------
    edges = np.vstack((F[:, [0, 1]],               # shape = (3F, 2)
                       F[:, [1, 2]],
                       F[:, [2, 0]])).astype(np.int64)

    edges_sorted = np.sort(edges, axis=1)          # undirected key

    # ---- 2. count edge occurrences ------------------------------------------
    uniq, inv, counts = np.unique(edges_sorted,
                                  axis=0,
                                  return_inverse=True,
                                  return_counts=True)

    # ---- 3. compute every edge’s UV length ----------------------------------
    diff = uv[edges[:, 0]] - uv[edges[:, 1]]
    lengths = np.linalg.norm(diff, axis=1)         # (3F,)

    # ---- 4. select edges that appear exactly once ---------------------------
    seam_mask = counts[inv] == 1
    return float(lengths[seam_mask].sum())




def count_zero_area_face_mesh(mesh_path: str, return_zero_face_idx: bool = False) -> int:
    """
    Count the number of zero-area faces in a mesh.
    """
    mesh = trimesh.load(mesh_path, force="mesh")
    
    uv_data = mesh.visual.uv  # shape: (num_vertices, 2)
    
    return count_zero_area_face(uv_data, mesh.faces, return_zero_face_idx=return_zero_face_idx  )

def count_zero_area_face(vertices: np.ndarray,
                       faces: np.ndarray,
                       tol: float = 0,
                       return_zero_face_idx: bool = False):
    """
    Detect whether any face is degenerate because at least two of its
    vertices coincide.

    Parameters
    ----------
    vertices : (N, 3) ndarray
        Mesh vertex positions.
    faces : (M, 3) ndarray
        Triangle indices.
    tol : float, optional
        Coordinate-equality tolerance (L2 distance). 1e-12 by default.

    Returns
    -------
    degenerate_exists : bool
        True if any face is degenerate.
    degenerate_mask : (M,) bool ndarray
        Boolean mask with True for degenerate faces (useful for filtering).
    """
    # --- 1. duplicate indices ----------------------------------------------
    dup_idx_mask = (
        (faces[:, 0] == faces[:, 1]) |
        (faces[:, 0] == faces[:, 2]) |
        (faces[:, 1] == faces[:, 2])
    )

    # --- 2. identical coordinates (distinct indices) -----------------------
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]


    dup01 = np.all(v0 == v1, axis=1)  # True if v0 and v1 coords match for a face
    dup02 = np.all(v0 == v2, axis=1)
    dup12 = np.all(v1 == v2, axis=1)
    dup_coord_mask = dup01 & dup02 & dup12  # shape (n_faces,)

    # combined result
    degenerate_mask = dup_idx_mask | dup_coord_mask
    count = int(degenerate_mask.sum())          # NEW: number of zero-area faces
    
    zero_face_idx = np.where(degenerate_mask)[0]

    if return_zero_face_idx:
        return count, zero_face_idx
    else:
        return count


def evaluate_mesh(obj_path, output_dir = None,  chart_num_as_filename=True, plot_overlapping_triangles=False):
    """
    Load a mesh with trimesh, then count how many UV charts it has.
    """
    if ".glb" in obj_path:
        mesh = trimesh.load(obj_path, process=False, force="mesh")
    else:
        mesh = trimesh.load(obj_path)
    # If no UVs exist, return 0
    if mesh.visual is None or mesh.visual.uv is None:
        return 0

    # uv_data is an (N,2) float array with one UV per vertex.
    uv_data = mesh.visual.uv  # shape: (num_vertices, 2)
    if len(uv_data) == 0:
        return 0

    # We want to index each unique UV coordinate so we can treat them like
    # the "uv indices" in an .obj file. Because these are floats, you might want
    # to do some rounding or hashing if there’s floating precision noise.
    uv_map = {}
    next_index = 0

    # Map from vertex index -> the integer "uv index"
    vertex_to_uv_index = [None] * len(mesh.vertices)

    for vidx, uv in enumerate(uv_data):
        # Convert floats to a tuple for dict keys
        key = (float(uv[0]), float(uv[1]))
        if key not in uv_map:
            uv_map[key] = next_index
            next_index += 1
        vertex_to_uv_index[vidx] = uv_map[key]

    # Create a "face_uv_indices" list
    faces_uv_indices = []
    for fidx in range(len(mesh.faces)):
        v0, v1, v2 = mesh.faces[fidx]
        faces_uv_indices.append([
            vertex_to_uv_index[v0],
            vertex_to_uv_index[v1],
            vertex_to_uv_index[v2]
        ])

    # Use union-find to group faces that share a full UV edge
    uf = UnionFind(len(faces_uv_indices))
    
    # We'll build an edge_map from (uvA, uvB) -> face_idx
    edge_map = {}

    for face_idx, uv_inds in enumerate(faces_uv_indices):
        # Each face is a triangle in trimesh, so we have 3 edges
        for i in range(3):
            a = uv_inds[i]
            b = uv_inds[(i + 1) % 3]
            if a > b:
                a, b = b, a
            edge = (a, b)
            if edge in edge_map:
                other_face_idx = edge_map[edge]
                uf.union(face_idx, other_face_idx)
            else:
                edge_map[edge] = face_idx

    # Count distinct connected components by counting unique "roots"
    roots = set()
    for i in range(len(faces_uv_indices)):
        roots.add(uf.find(i))

    # save uv layout with distortion
    charts = defaultdict(list)
    for face_idx in range(len(mesh.faces)):
        charts[uf.find(face_idx)].append(face_idx)
    chart_list = list(charts.values())
    
    all_distortions = []
    total_distortion = 0
    
    for i,chart in enumerate(chart_list):
        distortion = calculate_distortion_area(mesh.vertices, mesh.faces[chart], uv_data)
        all_distortions += [distortion]
        # if distortion > 2:
        #     print([calculate_3d_face_area(mesh.vertices, face) for face in mesh.faces[chart]])
        #     distortion = calculate_distortion_area(mesh.vertices, mesh.faces[chart], uv_data, multiplier=100000)
            
            

        total_distortion +=  distortion * len(chart)
        

    
    # count = overlap.compute_overlapping_triangles(torch.tensor(uv_data), torch.tensor(mesh.faces))
    # overlapping_triangles = {n for tup in count for n in tup} if plot_overlapping_triangles else None
    if(output_dir is not None):
        # output_dir = os.path.dirname(obj_path)
        if not os.path.exists(output_dir):
            if ".png" in output_dir or ".jpg" in output_dir:
                os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            else:
                os.makedirs(output_dir)
            
        
        if chart_num_as_filename:
            obj_uv_path = os.path.join(output_dir, f"{len(roots)}.png")
        elif ".png" in output_dir or ".jpg" in output_dir:
            obj_uv_path = output_dir
        else:
            obj_uv_path = os.path.join(output_dir, os.path.basename(obj_path).replace('.obj', '.png'))
            
        # save_uv_layout_no_normal_NUVO(mesh.vertices, mesh.faces, mesh.visual.uv, obj_uv_path, 
        #                                 image_size = 4096, save_image=True, overlapping_triangles=overlapping_triangles)
        # save_uv_layout_no_normal_NUVO(mesh.vertices, mesh.faces, mesh.visual.uv, obj_uv_path.split('.')[0] + '_angle.png',
        #                                 image_size = 4096, save_image=True, mode='angle', overlapping_triangles=overlapping_triangles)

        

        save_uv_layout_no_normal(mesh.vertices, mesh.faces, mesh.visual.uv, obj_uv_path, 
                                        image_size = 4096, save_image=True)
        save_uv_layout_no_normal(mesh.vertices, mesh.faces, mesh.visual.uv, obj_uv_path.split('.')[0] + '_angle.png',
                                        image_size = 4096, save_image=True, mode='angle')
    face_uvs = []
    
    for face in mesh.faces:
        face_uv = [uv_data[i] for i in face]
        face_uvs.append(face_uv)
    total_uv_area = sum(calculate_uv_face_area(uv) for uv in face_uvs)
    

    conformal_metric_median, conformal_metric_mean, conformal_metric_tile95 = conformal_metric(mesh)
    
    overall_distortion = calculate_distortion_area(mesh.vertices, mesh.faces, uv_data)
    tile95 = np.percentile(all_distortions, 95)
    
    
    L2_mesh, Linf_mesh, L2_invalid, Linf_invalid, num_invalid = stretch_metrics(mesh.vertices, mesh.faces, uv_data)
    
    result = {
        "num_charts": len(roots),
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "avg_distortion": total_distortion/len(mesh.faces),
        "max_distortion": max(all_distortions),
        "overall_distortion": overall_distortion,
        "distortion_tile95": tile95,
        "efficiency": total_uv_area,
        "overlap_count": -1 ,  # not available right now
        "total_distortion_0_1": calculate_distortion_area(mesh.vertices, mesh.faces, uv_data, distortion_normalized=True),
        "seam_length": uv_seam_length(mesh),
        "angular_distortion_mean": conformal_metric_mean,
        "angular_distortion_nuvo": conformal_metric_median,
        "angular_distortion_tile95": conformal_metric_tile95,
        "area_distortion_nuvo": calculate_distortion_area_nuvo(mesh.vertices, mesh.faces, uv_data),
        "zero_area_count": count_zero_area_face(uv_data, mesh.faces),
        "L2_mesh": float(L2_mesh),
        "Linf_mesh": float(Linf_mesh),
        "L2_invalid": bool(L2_invalid),
        "Linf_invalid": bool(Linf_invalid),
        "num_invalid": int(num_invalid) ,
    }
    
    return result



def copy_numeric_png(src, dst, prefix):
    f = next((f for f in os.listdir(src)
              if f.lower().endswith('.png') and re.fullmatch(r'\d+', os.path.splitext(f)[0])), None)
    if f:
        os.makedirs(dst, exist_ok=True)
        shutil.copy(os.path.join(src, f), os.path.join(dst, prefix + f))
        # print(f"Copied {f} to {dst}")





def main():

    # obj_path = "/ariesdv0/zhaoning/workspace/IUV/IUV/mesh_processing/output_meshes/random_obj_aa_udf/output/000a3d9fa4ff4c888e71e698694eb0b0/000a3d9fa4ff4c888e71e698694eb0b0_parts/xatlas_output/output.obj"
    
    # obj_path = "/ariesdv0/zhaoning/workspace/IUV/IUV/mesh_processing/output_meshes/random_obj_aa_udf_sub/output/aa6dea3384084a65986eda03c2cc11a7/aa6dea3384084a65986eda03c2cc11a7_parts/xatlas_output/output.obj"
    
    # charts = evaluate_mesh(obj_path)
    # copy_numeric_png("/ariesdv0/zhaoning/workspace/IUV/IUV/mesh_processing/output_meshes/random_obj_aa_udf/output/000a3d9fa4ff4c888e71e698694eb0b0/000a3d9fa4ff4c888e71e698694eb0b0_parts/xatlas_output", "/ariesdv0/zhaoning/workspace/IUV/IUV/mesh_processing/output_meshes/random_obj_aa_udf/output/000a3d9fa4ff4c888e71e698694eb0b0/000a3d9fa4ff4c888e71e698694eb0b0_parts/xatlas_output", "xatlas_")
    # print(f"Number of UV charts (islands): {charts}")
    
    obj_path = "/ariesdv0/zhaoning/workspace/IUV/lscm/libigl-example-project/mesh_processing/output_meshes/PartObjaverse-Tiny_mesh_all4/POT-24/00790c705e4c4a1fbc0af9bf5c9e9525/uv_all.obj"
    print(evaluate_mesh(obj_path))

    

if __name__ == "__main__":
    main()
