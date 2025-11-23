import os

import torch
import numpy as np
import time
from tqdm import tqdm
import trimesh

from preprocess_utils.partfield_official.run_PF import PFInferenceModel


import partuv
from partuv.preprocess import preprocess, save_results
import os
import os, sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pack.eval_charts import evaluate_mesh
from pack.pack import pack_mesh




def partuv_pipeline(args, pf_model=None, save_output=True):
    mesh_path = args.mesh_path
    output_path = args.output_path
    config_path = args.config_path
    hierarchy_path = args.hierarchy_path
    
    os.makedirs(output_path, exist_ok=True)
    
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    if hierarchy_path is None:
        
        """
        Preprocessing the mesh.

        The code does the following:
        - Load the mesh.
        - Remove existing UV layers, if any.
        - Merge overlapping vertices, and set epsilon to None to disable merging.
        - Fix non-2-manifold meshes, if present.
        - (Optional) export the mesh to an .obj file.
        - Run PartField to generate the hierarchical part tree.

        Notable parameters:
        - sample_on_faces and sample_batch_size: sample points on faces to obtain PartField features for part assignment.
        The larger sample_on_faces is, the more robust the part assignment is, but more points will be sampled and thus it takes more time.
        The larger sample_batch_size is, the less time it takes, but the more GPU memory it uses. Reduce it if you run out of memory.
        - merge_vertices_epsilon: epsilon for merging overlapping vertices; set to None to disable merging.
        - save_processed_mesh: set to False to disable exporting the processed mesh.
        - save_tree_file: set to False to disable saving the tree file for reproducibility.
        - output_path: path where the processed mesh and tree file will be saved.
        """
        
        mesh, tree_filename, tree_dict, preprocess_times = preprocess(mesh_path, pf_model, output_path, save_tree_file=True, save_processed_mesh=True, sample_on_faces=10, sample_batch_size=100_000, merge_vertices_epsilon=1e-7)
        V = mesh.vertices
        F = mesh.faces
        configPath = config_path
        print(f"F.shape after preprocessing: {F.shape}, V.shape: {V.shape}")
        final_parts, individual_parts = partuv.pipeline_numpy(
            V=V,
            F=F,
            tree_dict=tree_dict,
            configPath=configPath,
            threshold=1.25
        )
    else:
        """
        Use the provided hierarchy file and processed mesh instead of running Preprocessing.
        """
        tree_filename = hierarchy_path
        mesh_filename = mesh_path
        configPath = config_path

        final_parts, individual_parts = partuv.pipeline(
            tree_filename=tree_filename,
            mesh_filename=mesh_filename,
            configPath=configPath,
            threshold=1.25
        )
        
        
    if save_output:
        save_results(output_path, final_parts, individual_parts)
    print(f"Pipeline completed successfully!")
    print(f"Final parts: {final_parts.num_components} components")
    print(f"Final distortion: {final_parts.distortion}")
    print(f"Individual parts: {len(individual_parts)}")
    
    if final_parts.num_components > 0:
        uv_coords = final_parts.getUV()
        print(f"UV coordinates shape: {uv_coords.shape}")
        
        # You can also access individual components
        for i, component in enumerate(final_parts.components):
            print(f"Chart {i}: {component.F.shape[0]} faces, distortion: {component.distortion}")
        
    
    if args.pack_method in ["uvpackmaster", "blender"]:
        print(f"Starting to pack mesh with {args.pack_method} method")
        try:
            pack_mesh(output_path, uvpackmaster = args.pack_method == "uvpackmaster", save_visuals=args.save_visuals)
        except Exception as e:
            print(f"Error packing mesh with {args.pack_method} method: {e}")
            print(f"Skipping packing")
    
    # print(f"Evaluating final {output_path}/final_packed.obj")
    # metrics_after = evaluate_mesh(os.path.join(output_path, "final_packed.obj"))
    # print(metrics_after)
    
    # Access the UV coordinates
        
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run PartUV demo pipeline.")
    parser.add_argument("--mesh_path", type=str, default="/ariesdv0/zhaoning/workspace/partuv/swird_merge.obj", help="input mesh path")
    parser.add_argument("--config_path", "-cf", type=str, default="config/config.yaml", help="Config path.")
    parser.add_argument("--hierarchy_path", "-hp", type=str, default=None, help="(optional) Hierarchy path. If provided, it will be used directly in pipeline and skip the preprocessing (including PartField) entirely, make sure the mesh is preprocessed if the hierarchy file is provided.")
    parser.add_argument("--output_path", "-op", type=str, default=None, help="Output path.")
    parser.add_argument("--pack_method", "-pm", type=str, default="blender", choices=["blender", "uvpackmaster", "none"], help="Pack method.")
    parser.add_argument("--save_visuals", "-sv", action="store_true", default=False, help="Save visuals (such as) after packing. This will be ignored if pack_method is 'none'.")
    args = parser.parse_args()

    if args.output_path is None:
        mesh_name = os.path.basename(args.mesh_path).split(".")[0]
        args.output_path = os.path.join("./output", mesh_name)
        os.makedirs(args.output_path, exist_ok=True)
        
    pf_model = PFInferenceModel(device="cpu" if not torch.cuda.is_available() else "cuda")

    

    partuv_pipeline(args, pf_model, save_output=True)
    
    print("Done")


if __name__ == "__main__":
    main()