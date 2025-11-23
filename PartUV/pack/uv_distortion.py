#!/usr/bin/env python
import trimesh
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
# import networkx as nx
import torch
import math


import seaborn as sns
import matplotlib.colors as mcolors



def calculate_3d_face_area(vertices, face):
    """Calculate the area of a face in 3D space."""
    v0, v1, v2 = vertices[face]
    # Calculate face area using cross product
    area = np.linalg.norm(np.cross(v1 - v0, v2 - v0)) / 2
    return area

def calculate_total_area_vec(vertices, faces):
    """Calculate the total area of all faces in 3D space."""
    # Extract the vertices for each face
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Compute the cross product of the edges for all faces
    cross_products = np.cross(v1 - v0, v2 - v0)
    
    # Compute the area of each face
    areas = np.linalg.norm(cross_products, axis=1) / 2
    
    # Sum the areas of all faces
    total_area = np.sum(areas)
    return total_area


def calculate_uv_face_area(uvs):
    """Calculate the area of a face in UV space using the shoelace formula."""
    n = len(uvs)
    area = np.float64(0.0)
    for i in range(n):
        j = (i + 1) % n
        area += uvs[i][0] * uvs[j][1]
        area -= uvs[j][0] * uvs[i][1]
    return abs(area) / 2.0



def calculate_uv_areas_vec(face_uvs):
   # Prepare arrays for vectorized calculation
   n = len(face_uvs[0])
   shifts = np.roll(face_uvs, -1, axis=1)  # Shift vertices to get next vertex
   # Calculate areas using broadcasting
   areas = np.abs(np.sum(face_uvs[:,:,0] * shifts[:,:,1] - shifts[:,:,0] * face_uvs[:,:,1], axis=1)) / 2
   return areas



def color_from_area_distortion(ratio, cutoff=0.7):
    """
    Convert area-ratio to an (R, G, B) tuple using the Spectral_r palette
    (blue → green → yellow → red). Returned values are 0-255 ints.
    """
    _NORM = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # 256-step continuous colormap with the desired ordering
    _CMAP = sns.color_palette("coolwarm", as_cmap=True)
    # same distortion definition you used
    distortion = abs(1.0 - ratio) if ratio > 0 else 1.0
    distortion = min(distortion, cutoff) /cutoff

    # sample the colormap → floats in 0-1 → convert to 0-255 ints
    r, g, b, _ = _CMAP(_NORM(distortion))
    return int(r * 255), int(g * 255), int(b * 255)
    
    
    
    
def get_face_center_uv(uvs, scale):
    """Calculate the center point of a face in UV space."""
    center_x = sum(uv[0] for uv in uvs) / len(uvs)
    center_y = sum(uv[1] for uv in uvs) / len(uvs)
    return (int(center_x * scale), int((1 - center_y) * scale))

# def normalize_uv_coordinates(uvs):
#     """Normalize UV coordinates to [0, 1] range."""
#     min_u = min(uv[0] for uv in uvs)
#     max_u = max(uv[0] for uv in uvs)
#     min_v = min(uv[1] for uv in uvs)
#     max_v = max(uv[1] for uv in uvs)
    
#     uvs_normalized = []
#     for u, v in uvs:
#         u_norm = (u - min_u) / (max_u - min_u)
#         v_norm = (v - min_v) / (max_v - min_v)
#         uvs_normalized.append([u_norm, v_norm])
    
#     return uvs_normalized

def normalize_uv_coordinates(uvs: torch.Tensor):
    return (uvs - uvs.min(dim=0).values) / (uvs.max(dim=0).values - uvs.min(dim=0).values)

# @line_profiler.profile
def calculate_distortion(vertices, faces, uvs: torch.Tensor):
    """Calculate the overall average distortion per face."""
    # Convert inputs to numpy arrays
    vertices = np.array(vertices)
    # faces = [list(face) for face in faces]
    
    # Normalize UV coordinates
    uvs = normalize_uv_coordinates(uvs)
    uvs = np.array(uvs)
    
    # Compute total 3D area
    # total_3d_area = sum(calculate_3d_face_area(vertices, face) for face in faces)
    
    total_3d_area = calculate_total_area_vec(vertices, faces)
    # total_3d_area_2 = trimesh.Trimesh(vertices=vertices, faces=faces).area
    
    # Compute total UV area
    # face_uvs = []
    # for face in faces:
    #     face_uv = [uvs[i] for i in face]
    #     face_uvs.append(face_uv)
        
    face_uvs = uvs[faces]
    total_uv_area = sum(calculate_uv_areas_vec(face_uvs))
    
    # Calculate area scale factor
    area_scale_factor = total_3d_area / total_uv_area if total_uv_area > 0 else 1.0

    distortions = []
    
    vertices_array = vertices[faces]  # Your input array
    v0, v1, v2 = vertices_array[:, 0], vertices_array[:, 1], vertices_array[:, 2]
    area3d = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) / 2
    areauv = calculate_uv_areas_vec(face_uvs)
    
    ratio = (areauv * area_scale_factor) / area3d
    ratio[ratio > 1] = 1 / ratio[ratio > 1]
    distortion = np.abs(1.0 - ratio)
    distortions = distortion.tolist()
    
    
    # for face, face_uv in zip(faces, face_uvs):
    #     area_3d = calculate_3d_face_area(vertices, face)
    #     area_uv = calculate_uv_face_area(face_uv)
    #     # Avoid division by zero
    #     if area_3d == 0:
    #         ratio = 1.0
    #     else:
    #         ratio = (area_uv * area_scale_factor) / area_3d
    #     # Adjust ratio if greater than 1
    #     if ratio > 1:
    #         ratio = 1 / ratio
    #     distortion = abs(1.0 - ratio)
    #     distortions.append(distortion)
    
    # Calculate average distortion
    total_distortion = sum(distortions)  if distortions else 0.0
    return total_distortion


def normalize_to_unit_box(vertices: np.ndarray):
    """
    Normalize a mesh so its axis-aligned bounding box fits exactly in [-1, 1]^3.

    Parameters
    ----------
    vertices : (N, 3) ndarray
        Vertex positions.
    faces : (M, 3) ndarray
        Triangle indices (unchanged by normalization).

    Returns
    -------
    vertices_norm : (N, 3) ndarray
        Normalized vertex positions.
    faces : (M, 3) ndarray
        Same face array (returned for convenience).
    """
    vertices = vertices.astype(np.float64, copy=False)

    # Axis-aligned bounding box
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)

    # Translate to origin
    center   = (v_min + v_max) * 0.5
    shifted  = vertices - center

    # Uniform scale: largest half-extent → 1
    half_extent = (v_max - v_min).max() * 0.5
    if half_extent == 0:                      # degenerate mesh
        raise ValueError("Mesh has zero size along all axes.")
    scale = 1.0 / half_extent
    vertices_norm = shifted * scale           # now in [-1, 1]

    return vertices_norm


def calculate_distortion_area(vertices, faces, uvs, distortion_normalized=False):
    
    vertices = normalize_to_unit_box(vertices)
    
    # # Calculate total areas for normalization
    total_3d_area = sum(calculate_3d_face_area(vertices, face) for face in faces)
    
    # # Calculate total UV area
    face_uvs = []
    for face in faces:
        face_uv = [uvs[i] for i in face]
        face_uvs.append(face_uv)
    total_uv_area = sum(calculate_uv_face_area(uv) for uv in face_uvs)

    total_ratio =  total_uv_area / total_3d_area if total_3d_area > 0 else 1.0
    
    
    if total_ratio == 0:
        return 1.0
    
    distortions = []
    max_distortion = 0.0    
    skipped_faces = 0
    for i, (face, face_uv) in enumerate(zip(faces, face_uvs)):
        
        area_3d = calculate_3d_face_area(vertices, face)

        area_uv = calculate_uv_face_area(face_uv)

        if area_3d < total_3d_area / len(faces) / 10000 or area_3d == 0:
            skipped_faces += 1
            continue
        else:
            ratio = area_uv  / area_3d
        
        # ratio = area_uv  / area_3d if area_3d > 0 else 1.0

        distortion = ratio  / total_ratio
        if distortion == 0:
            continue

        if distortion_normalized:
            # distortion = distortion < 1.0 ? 1.0 - distortion : 1.0 - 1.0 / distortion;
            distortion =  1.0 - distortion if distortion < 1.0 else 1.0 - 1.0 / distortion
        else:
            # distortion =  distortion < 1 ? (1 - distortion) : (1 - 1 / distortion)
            distortion = distortion if distortion >= 1 else 1 / distortion
            
        if math.isnan(distortion):
            # distortion = 1000.0
            continue
        
        # if distortion > 10:
            # print(f"Max distortion at face {i} : {face} with distortion {distortion}, area_3d {area_3d}, area_uv {area_uv}")
        
        distortion = min(distortion, 10.0)
        
        distortions.append(distortion)
        max_distortion = max(max_distortion, distortion)
        # if distortion > 10:
        #     print(f"Max distortion at face {face}")
            
            
    total_distortion = sum(distortions) / len(distortions) if distortions else 0.0
    return total_distortion


def conformal_metric(mesh: trimesh.Trimesh) -> float:
    return calculate_angle_distortion(mesh.vertices, mesh.faces, mesh.visual.uv)
    
    
def calculate_angle_distortion(vertices, faces, uvs):
    # Ensure UVs are present
    uv = uvs
    if uv is None:
        raise ValueError("Mesh contains no UV coordinates (mesh.visual.uv is None)")

    # Face indices
    faces = faces  # (F, 3)

    # Gather vertex positions and UVs per face
    P = vertices[faces]      # (F, 3, 3)
    UV = uv[faces]                # (F, 3, 2)

    # Compute edge differences in UV domain
    uv0, uv1, uv2 = UV[:, 0], UV[:, 1], UV[:, 2]  # each (F,2)
    du1, dv1 = uv1[:, 0] - uv0[:, 0], uv1[:, 1] - uv0[:, 1]
    du2, dv2 = uv2[:, 0] - uv0[:, 0], uv2[:, 1] - uv0[:, 1]

    # Compute world-space edge vectors
    P0, P1, P2 = P[:, 0], P[:, 1], P[:, 2]        # each (F,3)
    D = P1 - P0
    E = P2 - P0

    # Inverse of 2×2 UV Jacobian per face
    det = du1 * dv2 - dv1 * du2
    # Avoid degenerate or zero-area UV triangles
    valid = np.abs(det) > 1e-12
    if not np.any(valid):
        # raise ValueError("All UV triangles are degenerate")
        return 0, 0

    inv_det = 1.0 / det[valid]
    inv00 =  dv2[valid] * inv_det
    inv01 = -dv1[valid] * inv_det
    inv10 = -du2[valid] * inv_det
    inv11 =  du1[valid] * inv_det

    # Compute tangent (e_u) and bitangent (e_v) vectors
    Dv = D[valid]
    Ev = E[valid]
    e_u = inv00[:, None] * Dv + inv01[:, None] * Ev
    e_v = inv10[:, None] * Dv + inv11[:, None] * Ev

    # Compute cosine of angle between e_u and e_v for each face
    dot = np.einsum('ij,ij->i', e_u, e_v)
    norm = np.linalg.norm(e_u, axis=1) * np.linalg.norm(e_v, axis=1)
    cosines = np.abs(dot / norm)

    # Median cosine and conformal metric
    median_cos = np.median(cosines)
    mean_cos = np.mean(cosines)
    
    tile95_cos = np.percentile(cosines, 95)
    
    return 1.0 - median_cos, 1.0 - mean_cos, 1.0 - tile95_cos




def stretch_metrics(vertices, faces, uvs):
    """
    Compute texture‑stretch metrics 𝐿2(𝑀) and 𝐿∞(𝑀) together with
    their per‑triangle values.

    Parameters
    ----------
    vertices : (N, 3) float
        3‑D vertex coordinates q_i.
    faces    : (F, 3) int
        Indices of each triangle (counter‑clockwise in UV space).
    uvs      : (N, 2) float
        2‑D texture coordinates p_i = (s_i, t_i).

    Returns
    -------
    L2_mesh  : float          # area‑weighted RMS stretch 𝐿2(𝑀)
    Linf_mesh: float          # worst‑case stretch      𝐿∞(𝑀)
    L2_T     : (F,) ndarray   # per‑triangle 𝐿2(𝑇)
    Linf_T   : (F,) ndarray   # per‑triangle 𝐿∞(𝑇)
    """
    # Gather triangle corners
    q1, q2, q3 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    p1, p2, p3 = uvs[faces[:, 0]], uvs[faces[:, 1]], uvs[faces[:, 2]]

    s1, t1 = p1[:, 0], p1[:, 1]
    s2, t2 = p2[:, 0], p2[:, 1]
    s3, t3 = p3[:, 0], p3[:, 1]

    # 2 × parametric area  (denominator of Eq. (33))
    A2 = (s2 - s1) * (t3 - t1) - (s3 - s1) * (t2 - t1)
    area3D = 0.5 * np.linalg.norm(np.cross(q2 - q1, q3 - q1), axis=1)
    
    # Normalize UVs so that the total area in UV space matches the total area in 3D
    # Compute total 3D area and total UV area
    area3D_total = np.sum(area3D)
    area2D = 0.5 * np.abs(A2)
    area2D_total = np.sum(area2D)
    if area2D_total > 0:
        scale = np.sqrt(area3D_total / area2D_total)
        uvs = uvs * scale
        # Recompute p1, p2, p3 and A2 with scaled UVs
        p1, p2, p3 = uvs[faces[:, 0]], uvs[faces[:, 1]], uvs[faces[:, 2]]
        s1, t1 = p1[:, 0], p1[:, 1]
        s2, t2 = p2[:, 0], p2[:, 1]
        s3, t3 = p3[:, 0], p3[:, 1]
        A2 = (s2 - s1) * (t3 - t1) - (s3 - s1) * (t2 - t1)
    
    denom = A2
    # denom = 2.0 * A2



    with np.errstate(divide='ignore', invalid='ignore'):
        # Partial derivatives S_s and S_t  (Eq. (33))
        Ss = (q1 * (t2 - t3)[:, None] +
              q2 * (t3 - t1)[:, None] +
              q3 * (t1 - t2)[:, None]) / denom[:, None]

        St = (q1 * (s3 - s2)[:, None] +
              q2 * (s1 - s3)[:, None] +
              q3 * (s2 - s1)[:, None]) / denom[:, None]

        a = np.sum(Ss * Ss, axis=1)     # S_s·S_s
        b = np.sum(Ss * St, axis=1)     # S_s·S_t
        c = np.sum(St * St, axis=1)     # S_t·S_t

        # Per‑triangle norms (Eq. (24))
        L2_T   = np.sqrt((a + c) * 0.5)
        Linf_T = np.sqrt(0.5 * (a + c + np.sqrt((a - c)**2 + 4.0 * b**2)))

    # Infinite stretch for flipped or degenerate UVs
    # invalid  = (A2 < 0) | np.isnan(A2) | np.isinf(A2)
    
    invalid  = (A2 < 0) 
    L2_T[invalid]   = np.inf
    Linf_T[invalid] = np.inf
    
    # Remove invalid triangles from L2_T and Linf_T
    # L2_T   = L2_T[~invalid]
    # Linf_T = Linf_T[~invalid]
    
    # if invalid.any():
    #     print(f"Invalid triangles: {invalid.sum()}")

    num_invalid = invalid.sum()
    # 3‑D triangle areas for weighting (Eq. (31))
    # area3D = area3D[~invalid]

    L2_mesh  = np.sqrt(np.sum(L2_T**2 * area3D) / np.sum(area3D))
    Linf_mesh = np.max(Linf_T)
    
    L2_invalid = np.isnan(L2_mesh) or np.isinf(L2_mesh)
    Linf_invalid = np.isnan(Linf_mesh) or np.isinf(Linf_mesh)
    

    return L2_mesh, Linf_mesh, L2_invalid, Linf_invalid, num_invalid



def calculate_distortion_area_nuvo(vertices, faces, uvs):
    
    # # Calculate total areas for normalization
    total_3d_area = sum(calculate_3d_face_area(vertices, face) for face in faces)
    
    # # Calculate total UV area
    face_uvs = []
    for face in faces:
        face_uv = [uvs[i] for i in face]
        face_uvs.append(face_uv)
    total_uv_area = sum(calculate_uv_face_area(uv) for uv in face_uvs)

    total_ratio =  total_uv_area / total_3d_area if total_3d_area > 0 else 1.0
    
    distortions = []
    max_distortion = 0.0    
    for face, face_uv in zip(faces, face_uvs):
        
        area_3d = calculate_3d_face_area(vertices, face)
        area_uv = calculate_uv_face_area(face_uv)
        ratio = area_uv  / area_3d if area_3d > 0 else 1.0
        
        distortions.append(ratio)
        max_distortion = max(max_distortion, ratio)
        # if distortion > 200000:
        #     print(f"Max distortion at face {face}")
    median_distortion = np.median(distortions) if distortions else 0.0
    # total_distortion = sum(distortions) / len(distortions) if distortions else 0.0
    nuvo_distortion = np.median( np.abs(distortions - median_distortion))
    
    return 1 - nuvo_distortion


def remap_uv_strip_to_grid(uv, grid_cols):
    """
    Remap UVs laid out as a horizontal strip of tiles into a (grid_rows × grid_cols) grid.

    Parameters
    ----------
    uv : (N, 2) array-like
        Original UVs.  Only the *u* coordinate can exceed 1; v is already in [0, 1).
    grid_cols : int
        Number of tiles per row in the target grid (e.g. 4 for a 4×2 grid).

    Returns
    -------
    uv_out : (N, 2) ndarray
        UVs in the new grid.  Both coordinates may now be > 1.
    """
    uv      = np.asarray(uv, dtype=np.float64)
    u, v    = uv[:, 0], uv[:, 1]

    tile    = np.floor(u).astype(int)   # integer tile index along the original strip
    frac_u  = u - tile                  # keep in-tile fraction in x
    col     = tile % grid_cols          # column in the new grid
    row     = tile // grid_cols         # row   in the new grid

    u_out   = frac_u + col
    v_out   = v       + row
    return np.column_stack((u_out, v_out))



def split_faces_by_atlas(uv, faces, num_atlases=8):
    """
    Partition faces into `num_atlases` subsets, one per UV tile (atlas).

    Parameters
    ----------
    uv : (V, 2) ndarray
        UV coordinates for each vertex, with atlases laid out along +u.
    faces : (F, 3) ndarray (int)
        Triangle indices into `uv`.
    num_atlases : int, default 8
        How many atlases/tiles exist along the strip.

    Returns
    -------
    subsets : list of (Ni, 3) ndarrays
        `subsets[i]` contains all faces that lie entirely in atlas *i*,
        where *i* ∈ {0,…,num_atlases−1}.  Faces crossing atlas borders
        are ignored.
    """
    atlas_id      = np.floor(uv[:, 0]).astype(int)   # per-vertex atlas
    face_atlas_id = atlas_id[faces]                  # (F, 3) atlas IDs

    same_tile     = np.all(face_atlas_id ==
                           face_atlas_id[:, [0]], axis=1)  # all 3 equal?
    face_indices  = np.where(same_tile)[0]           # indices of valid faces
    atlas_face_id = face_atlas_id[same_tile, 0]     # atlas number per face

    subsets = [face_indices[atlas_face_id == i] for i in range(num_atlases)]
    return subsets


def save_uv_layout_no_normal(vertices, faces,  uvs, filepath=None, image_size=1024, mode='area', draw_text=False, return_map_image=False, save_image=False, input_image = None, overlapping_triangles=None):
    """Save UV layout with distortion visualization and numeric values."""
    # Load mesh
    # mesh = trimesh.load(mesh_path)
    # Create image
    faces_subsets = None
    if input_image:
        img = input_image
    elif max(uvs[:,0]) > 1:
        # we are handling NUVO UVs
        faces_subsets = split_faces_by_atlas(uvs, faces)
        
        num_cols = 4
        num_rows = 8 // num_cols
        uvs = remap_uv_strip_to_grid(uvs, num_cols)
        
        img = Image.new('RGBA', (image_size*num_cols, image_size*num_rows),  (233,234,244,255))
        
    else:
        img = Image.new('RGBA', (image_size, image_size),  (233,234,244,255))
    draw = ImageDraw.Draw(img)
    
    # uvs = normalize_uv_coordinates(uvs)
    # # Calculate total areas for normalization
    total_3d_area = sum(calculate_3d_face_area(vertices, face) for face in faces)
    
    # # Calculate total UV area
    face_uvs = []
    for face in faces:
        face_uv = [uvs[i] for i in face]
        face_uvs.append(face_uv)
    total_uv_area = sum(calculate_uv_face_area(uv) for uv in face_uvs)
    
    
    
    area_scale_factor = total_3d_area / total_uv_area if total_uv_area > 0 else 1.0
    
    
    
    scale = image_size - 1
    # if max(uvs[:,0]) > 1:
    #     num_cols = 4
    #     num_rows = 8 // num_cols
    #     scale_x = image_size * num_cols - 1
    #     scale_y = image_size * num_rows - 1
    # else:
    #     scale_x = image_size - 1
    #     scale_y = image_size - 1
    
    # Store face data for two-pass rendering
    face_data = []


    
    for face_idx, (face, face_uv) in enumerate(zip(faces, face_uvs)):
        # Convert UV coordinates to image space
        uvs_px = [(int(uv[0] * scale), int((1 - uv[1]) * scale)) for uv in face_uv]
        
        if overlapping_triangles is not None and face_idx in overlapping_triangles:
            color = color_from_area_distortion(1)
            
        elif mode == 'area':
            area_3d = calculate_3d_face_area(vertices, face)
            area_uv = calculate_uv_face_area(face_uv)
            ratio = (area_uv * area_scale_factor) / area_3d if area_3d > 0 else 1.0
            if ratio > 1:
                color_ratio = 1/ratio
            else:
                color_ratio = ratio
            color = color_from_area_distortion(color_ratio)
            
            
            distortion = ratio if ratio > 1 else 1 / ratio
            # distortion = abs(1.0 - ratio)
            
            
        elif mode == 'angle':
            distor_result = calculate_angle_distortion(vertices, face.reshape(1,3), uvs)
            distortion = distor_result[1]
            
            color = color_from_area_distortion(distortion)
            
            
            
            
        elif type(mode) == tuple: 
            distortion = None
            color = tuple([int(c * 255) for c in mode])
            assert draw_text == False, "Cannot draw text with custom color"
        
        # Draw filled polygon
        draw.polygon(uvs_px, fill=color)
        
        # Draw edges in white
        for i in range(len(uvs_px)):
            start = uvs_px[i]
            end = uvs_px[(i + 1) % len(uvs_px)]
            draw.line([start, end], fill=(255, 255, 255, 196), width=1)
        
        # Store face data for text rendering
        center = get_face_center_uv(face_uv, scale)
        face_data.append((center, distortion))
    
    # if len(count) > 0:
    #     for face in count:
    #         face_uv = [uvs[i] for i in faces[face]]
    #         uvs_px = [(int(uv[0] * scale), int((1 - uv[1]) * scale)) for uv in face_uv]
    #         draw.polygon(uvs_px, fill=(255, 0, 0))
    #         for i in range(len(uvs_px)):
    #             start = uvs_px[i]
    #             end = uvs_px[(i + 1) % len(uvs_px)]
    #             draw.line([start, end], fill='white', width=1)
            
    #         center = get_face_center_uv(face_uv, scale)
    #         face_data.append((center, -1))
    
    # Convert to RGBA for anti-aliased text
    img = img.convert('RGBA')
    draw = ImageDraw.Draw(img)
    
    # Try to load font
    try:
        try:
            font = ImageFont.truetype("Arial.ttf", 14)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                font = ImageFont.load_default()
    except ImportError:
        font = ImageFont.load_default()
    
    # Second pass: draw text
    if draw_text:
        for i, (center, distortion )in enumerate(face_data):
            text = f"{distortion:.2f}"
            
            try:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except:
                text_width, text_height = draw.textsize(text, font=font)
            
            x = center[0] - text_width // 2
            y = center[1] - text_height // 2
            
            # Draw text outline
            outline_positions = [
                (x-1, y-1), (x+1, y-1), (x-1, y+1), (x+1, y+1),
                (x-1, y), (x+1, y), (x, y-1), (x, y+1)
            ]
            # for ox, oy in outline_positions:
            #     draw.text((ox, oy), text, fill=(0, 0, 0, 255), font=font)
            
            # Draw main text
            draw.text((x, y), str(text), fill=(255, 255, 255, 255), font=font)
 
    if filepath and save_image:
        img.save(filepath)
    if return_map_image:
        return img
    else:
        return 




def save_uv_layout_no_normal_NUVO(vertices, faces,  uvs, filepath=None, image_size=1024, mode='area', draw_text=False, return_map_image=False, save_image=False, input_image = None, overlapping_triangles=None):
    """Save UV layout with distortion visualization and numeric values."""
    # Load mesh
    # mesh = trimesh.load(mesh_path)
    # Create image
    faces_subsets = None
    if input_image:
        img = input_image
    elif max(uvs[:,0]) > 1:
        # we are handling NUVO UVs
        faces_subsets = split_faces_by_atlas(uvs, faces)
        
        num_cols = 4
        num_rows = 8 // num_cols
        uvs = remap_uv_strip_to_grid(uvs, num_cols)
        
        img = Image.new('RGBA', (image_size*num_cols, image_size*num_rows),  (233,234,244,255))
        
    else:
        img = Image.new('RGBA', (image_size, image_size),  (233,234,244,255))
    draw = ImageDraw.Draw(img)
    
    # uvs = normalize_uv_coordinates(uvs)
    # # Calculate total areas for normalization
    # total_3d_area = sum(calculate_3d_face_area(vertices, face) for face in faces)
    total_3d_area_subset = []
    

    
    
    # # Calculate total UV area
    face_uvs = []
    for face in faces:
        face_uv = [uvs[i] for i in face]
        face_uvs.append(face_uv)
        
        
    # total_uv_area = sum(calculate_uv_face_area(uv) for uv in face_uvs)
    # total_uv_area_subsets = [sum(calculate_uv_face_area(uv) for uv in face_uvs) for face_uvs in faces_subsets]
    
    total_uv_area_subsets = []
    total_area_scale_factors = []
    for face_indices in faces_subsets:
        
        total_3d_area_subset.append(sum(calculate_3d_face_area(vertices, faces[face_idx]) for face_idx in face_indices))
        face_uvs_subset = [face_uvs[face_idx] for face_idx in face_indices]
        total_uv_area_subsets.append(sum(calculate_uv_face_area(uv) for uv in face_uvs_subset))
        total_area_scale_factors.append(total_3d_area_subset[-1] / total_uv_area_subsets[-1] if total_uv_area_subsets[-1] > 0 else 1.0)
        # area_scale_factor = total_3d_area / total_uv_area if total_uv_area > 0 else 1.0
    
    
    scale = image_size - 1

    face_data = []


    def get_atlas_id(face_uv):
        return int(np.floor(face_uv[0][ 1])* num_cols + np.floor(face_uv[0][0]))
    
    for face_idx, (face, face_uv) in enumerate(zip(faces, face_uvs)):
        # Convert UV coordinates to image space
        uvs_px = [(int(uv[0] * scale), int((num_rows - uv[1]) * scale)) for uv in face_uv]
        
        if overlapping_triangles is not None and face_idx in overlapping_triangles:
            color = color_from_area_distortion(1)
            
        elif mode == 'area':
            area_3d = calculate_3d_face_area(vertices, face)
            area_uv = calculate_uv_face_area(face_uv)

            ratio = (area_uv * total_area_scale_factors[get_atlas_id(face_uv)]) / area_3d if area_3d > 0 else 1.0
            if ratio > 1:
                color_ratio = 1/ratio
            else:
                color_ratio = ratio
            color = color_from_area_distortion(color_ratio)
            
            
            distortion = ratio if ratio > 1 else 1 / ratio
            # distortion = abs(1.0 - ratio)
            
            
        elif mode == 'angle':
            _,distortion = calculate_angle_distortion(vertices, face.reshape(1,3), uvs)
            
            color = color_from_area_distortion(distortion)
            
            
            
            
        elif type(mode) == tuple: 
            distortion = None
            color = tuple([int(c * 255) for c in mode])
            assert draw_text == False, "Cannot draw text with custom color"
        
        # Draw filled polygon
        draw.polygon(uvs_px, fill=color)
        
        # Draw edges in white
        for i in range(len(uvs_px)):
            start = uvs_px[i]
            end = uvs_px[(i + 1) % len(uvs_px)]
            draw.line([start, end], fill=(255, 255, 255, 196), width=1)
        
        # Store face data for text rendering
        center = get_face_center_uv(face_uv, scale)
        face_data.append((center, distortion))
    
    # if len(count) > 0:
    #     for face in count:
    #         face_uv = [uvs[i] for i in faces[face]]
    #         uvs_px = [(int(uv[0] * scale), int((1 - uv[1]) * scale)) for uv in face_uv]
    #         draw.polygon(uvs_px, fill=(255, 0, 0))
    #         for i in range(len(uvs_px)):
    #             start = uvs_px[i]
    #             end = uvs_px[(i + 1) % len(uvs_px)]
    #             draw.line([start, end], fill='white', width=1)
            
    #         center = get_face_center_uv(face_uv, scale)
    #         face_data.append((center, -1))
    
    # Convert to RGBA for anti-aliased text
    img = img.convert('RGBA')
    draw = ImageDraw.Draw(img)
    
    # Try to load font
    try:
        try:
            font = ImageFont.truetype("Arial.ttf", 14)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                font = ImageFont.load_default()
    except ImportError:
        font = ImageFont.load_default()
    
    # Second pass: draw text
    if draw_text:
        for i, (center, distortion )in enumerate(face_data):
            text = f"{distortion:.2f}"
            
            try:
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except:
                text_width, text_height = draw.textsize(text, font=font)
            
            x = center[0] - text_width // 2
            y = center[1] - text_height // 2
            
            # Draw text outline
            outline_positions = [
                (x-1, y-1), (x+1, y-1), (x-1, y+1), (x+1, y+1),
                (x-1, y), (x+1, y), (x, y-1), (x, y+1)
            ]
            # for ox, oy in outline_positions:
            #     draw.text((ox, oy), text, fill=(0, 0, 0, 255), font=font)
            
            # Draw main text
            draw.text((x, y), str(text), fill=(255, 255, 255, 255), font=font)
 
    if filepath and save_image:
        img.save(filepath)
    if return_map_image:
        return img
    else:
        return 


def save_uv_layout_with_distortion(vertices, faces,  uvs, filepath=None, image_size=1024, mode='area', draw_text=False, return_map_image=False, save_image=False):
    """Save UV layout with distortion visualization and numeric values."""
    # Load mesh
    # mesh = trimesh.load(mesh_path)
    # Create image
    uvs = normalize_uv_coordinates(uvs)
    return save_uv_layout_no_normal(vertices, faces, uvs, filepath, image_size, mode, draw_text, return_map_image, save_image)



def scale_charts(charts):
    total_area = sum(c[:, 0].max() * c[:, 1].max() for c in charts)




def save_uv_layout_with_packing(components, chart_uvs_list, filepath=None, image_size=1024, mode='area', draw_text=False, return_map_image=False, packing=True, save_image=False):
    """
    Similar to your original save function, but first packs the charts into [0,1]^2,
    and then uses the packed UVs to draw.
    """


    packed_charts = chart_uvs_list

    image = None
    for i, (comp, chart) in enumerate(zip(components, packed_charts)):
        if type(mode) == list:
            draw_mode = mode[i]
        else:
            draw_mode = mode
              
        image = save_uv_layout_no_normal(comp.vertices, comp.faces, chart, image_size=image_size, mode=draw_mode, draw_text=draw_text, return_map_image=True, input_image=image)#, filepath="test.png", save_image=True)
        
    if filepath and save_image:
        image.save(filepath)
    return packed_charts
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate UV distortion visualization')
    parser.add_argument('--mesh_path', help='Path to input mesh file', default="/ariesdv0/zhaoning/workspace/IUV/lscm/libigl-example-project/mesh_processing/output_meshes/POT-mini2/POT-mini/0c3ca2b32545416f8f1e6f0e87def1a6/uv_all.obj")
    parser.add_argument('--output_path', help='Path to output image', default="test.png")
    parser.add_argument('--size', type=int, default=1024, help='Output image size')
    parser.add_argument('--mode', choices=['area', 'angle'], default='area', 
                        help='Distortion measurement mode')
    parser.add_argument('--draw-text', action='store_true', 
                        help='Draw distortion values on faces')
    
    args = parser.parse_args()
    
    # save_uv_layout_with_distortion(
    #     args.mesh_path,
    #     args.output_path,
    #     image_size=args.size,
    #     mode=args.mode,
    #     draw_text=args.draw_text
    # )

    mesh = trimesh.load(args.mesh_path)
    uvs = mesh.visual.uv
    faces = mesh.faces
    vertices = mesh.vertices
    # calculate_angle_distortion(vertices, faces, uvs)
    save_uv_layout_no_normal(vertices, faces, uvs, filepath=args.output_path, image_size=args.size, mode='angle', save_image=True)
    
    
    
    
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
    
    for chart in chart_list:
        distortion = calculate_distortion_area(mesh.vertices, mesh.faces[chart], uv_data)
        all_distortions += [distortion]

        total_distortion +=  distortion * len(chart)
        

    
    count = overlap.compute_overlapping_triangles(torch.tensor(uv_data), torch.tensor(mesh.faces))
    overlapping_triangles = {n for tup in count for n in tup} if plot_overlapping_triangles else None
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
                                        image_size = 4096, save_image=True, overlapping_triangles=overlapping_triangles)
        save_uv_layout_no_normal(mesh.vertices, mesh.faces, mesh.visual.uv, obj_uv_path.split('.')[0] + '_angle.png',
                                        image_size = 4096, save_image=True, mode='angle', overlapping_triangles=overlapping_triangles)
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
        "overlap_count": len(count),
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
