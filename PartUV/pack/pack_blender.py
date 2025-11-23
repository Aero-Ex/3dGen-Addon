import bpy
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Blender UV Packing Script')
    parser.add_argument('--mesh_folder', type=str, required=True, help='Path to mesh folder containing final_components.obj')
    return parser.parse_args()


def import_obj(filepath, new_name="my_object"):
    # Load the OBJ file
    bpy.ops.wm.obj_import(filepath=filepath)
    
    # The newly imported object will be selected automatically,
    # so we can retrieve it with:
    obj = bpy.context.selected_objects[0]
    
    # Now rename the object
    obj.name = new_name
    return obj

#     in_path = "/home/wzn/workspace/partuv/00aee5c2fef743d69421bb642d446a5b/final_components.obj"
def pack_blender(mesh_folder, out_path=None):
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')
    # Remove all objects in the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Set paths using the provided mesh_folder
    in_path = os.path.join(mesh_folder, 'final_components.obj')
    out_path = os.path.join(mesh_folder, f'final_packed.obj') if out_path is None else out_path

    print(in_path, out_path)


    if os.path.exists(in_path):
        obj = import_obj(in_path, "final_components")
    else:
        print(f"File not found: {in_path}")
        exit()

    # in_path = "./00aee5c2fef743d69421bb642d446a5b/final_components.obj"    
    # obj = import_obj(in_path, "00aee5c2fef743d69421bb642d446a5b")


    bpy.ops.object.select_all(action='SELECT')
        
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # Select all UVs and pack islands
    bpy.ops.uv.select_all(action='SELECT')
    # bpy.ops.uv.average_islands_scale()

    bpy.ops.uv.pack_islands(margin=0.001)
    # save the obj with packed uv


    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.wm.obj_export(filepath=out_path, export_selected_objects=True, export_materials=False, export_normals=False)

if __name__ == "__main__":
    args = parse_args()

    # mesh_folder = "output/f16k-c12/"
    mesh_folder = args.mesh_folder
    pack_blender(mesh_folder)