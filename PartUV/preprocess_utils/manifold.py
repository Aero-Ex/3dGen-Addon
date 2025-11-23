#handles non-manifold meshes

import argparse
from collections import defaultdict
import numpy as np
import trimesh


def split_nonmanifold_edges(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Return a copy of *mesh* in which every edge is incident
    to at most two faces (2-manifold).
    """
    mesh = mesh.copy()
    V_orig = mesh.vertices.copy()
    F = mesh.faces

    # Build a map: sorted edge -> list[face indices] that contain it
    edge_to_faces = defaultdict(list)
    for fi, tri in enumerate(F):
        for a, b in ((tri[0], tri[1]),
                     (tri[1], tri[2]),
                     (tri[2], tri[0])):
            edge_to_faces[tuple(sorted((a, b)))].append(fi)

    vertices = V_orig.tolist()          # mutable growing list
    non_manifold_edges_count = 0
    for edge, faces in edge_to_faces.items():
        if len(faces) <= 2:
            continue                    # already manifold
        
        non_manifold_edges_count += 1
        # print(f"edge is not manifold: {edge}, faces: {faces}")
        v1, v2 = edge
        # Keep the first two faces untouched; fix the rest
        for fi in faces[2:]:
            # duplicate the two vertices
            v1_new = len(vertices)
            vertices.append(V_orig[v1])
            v2_new = len(vertices)
            vertices.append(V_orig[v2])

            face = F[fi].copy()
            # Replace occurrences of the old indices with the new ones
            face = np.where(face == v1, v1_new, face)
            face = np.where(face == v2, v2_new, face)
            F[fi] = face

    return trimesh.Trimesh(vertices=np.asarray(vertices),
                           faces=F,
                           process=False), non_manifold_edges_count   # keep indices as-is

def fix_mesh(mesh_path: str):
    mesh = trimesh.load(mesh_path, process=False)
    fixed, non_manifold_edges_count = split_nonmanifold_edges(mesh)
    if non_manifold_edges_count > 0:
        fixed.export(mesh_path)



def fix_mesh_trimesh(mesh: trimesh.Trimesh):
    fixed, non_manifold_edges_count = split_nonmanifold_edges(mesh)
    if non_manifold_edges_count > 0:
        return fixed
    return mesh
    
def main():
    mesh_path = "/ariesdv0/zhaoning/workspace/IUV/lscm/libigl-example-project/mesh_processing/output_meshes/PartObjaverse-Tiny_mesh_all2/output/2dd2980caf87450d9ef7e42adb43f2ca/seg/2dd2980caf87450d9ef7e42adb43f2ca_seg.obj"

    mesh = trimesh.load(mesh_path, process=False)
    fixed, non_manifold_edges_count = split_nonmanifold_edges(mesh)

    out_path = (mesh_path.rsplit(".", 1)[0] + "_fixed.obj")
    
    fixed.export(mesh_path)
    print(f"Non-manifold edges split.  Saved to: {out_path}")


if __name__ == "__main__":
    main()
