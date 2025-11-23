from uv_distortion import *
import numpy as np
import sys
import subprocess
import matplotlib.pyplot as plt
import os
import argparse
from scipy.spatial import cKDTree
import time

def generate_unique_colors(num_ids):
    used = set()
    colors = []
    while len(colors) < num_ids:
        c = tuple(np.random.rand(3))
        if c not in used:
            used.add(c)
            colors.append(np.array(c))
    for i, color in enumerate(colors):
        print(f"Color {i}: {color}")
    return colors

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
