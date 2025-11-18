"""
Multi-image operators and UI list for TRELLIS
"""

import bpy
from bpy.types import Operator, UIList
import os


class TRELLIS_UL_ImageList(UIList):
    """UI List for displaying multiple images with previews"""

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            if item.filepath:
                # Try to load and show image preview
                try:
                    img = bpy.data.images.get(os.path.basename(item.filepath))
                    if not img and os.path.exists(item.filepath):
                        img = bpy.data.images.load(item.filepath, check_existing=True)
                    
                    if img:
                        # Ensure preview is generated
                        img.preview_ensure()
                        # Show preview icon
                        layout.template_icon(icon_value=img.preview.icon_id, scale=3.0)
                except:
                    pass
                
                # Show filename
                filename = os.path.basename(item.filepath)
                layout.label(text=filename)
                
                # Show check if file exists
                if os.path.exists(item.filepath):
                    layout.label(text="", icon='CHECKMARK')
                else:
                    layout.label(text="", icon='ERROR')
            else:
                layout.label(text="(Empty)", icon='X')
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.label(text="", icon='IMAGE_DATA')


class TRELLIS_OT_AddMultiImage(Operator):
    """Add image(s) to multi-image list"""
    bl_idname = "trellis.add_multi_image"
    bl_label = "Add Image(s)"
    bl_description = "Add one or more images to the list"
    bl_options = {'REGISTER', 'UNDO'}

    filepath: bpy.props.StringProperty(
        name="Image File",
        description="Select image file(s)",
        subtype='FILE_PATH',
    )
    
    directory: bpy.props.StringProperty(
        subtype='DIR_PATH',
    )
    
    files: bpy.props.CollectionProperty(
        type=bpy.types.OperatorFileListElement,
        options={'HIDDEN', 'SKIP_SAVE'},
    )

    filter_image: bpy.props.BoolProperty(
        default=True,
        options={'HIDDEN'},
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        props = context.scene.trellis_props
        
        added_count = 0
        
        # Handle multiple file selection
        if self.files:
            for file_elem in self.files:
                filepath = os.path.join(self.directory, file_elem.name)
                if os.path.exists(filepath):
                    item = props.multi_images.add()
                    item.filepath = filepath
                    added_count += 1
        # Handle single file selection
        elif self.filepath:
            if os.path.exists(self.filepath):
                item = props.multi_images.add()
                item.filepath = self.filepath
                added_count += 1
        
        if added_count == 0:
            self.report({'WARNING'}, "No valid files selected")
            return {'CANCELLED'}
        
        # Set last added as active
        props.multi_images_index = len(props.multi_images) - 1
        
        self.report({'INFO'}, f"Added {added_count} image(s)")
        return {'FINISHED'}


class TRELLIS_OT_RemoveMultiImage(Operator):
    """Remove image from multi-image list"""
    bl_idname = "trellis.remove_multi_image"
    bl_label = "Remove Image"
    bl_description = "Remove selected image from the list"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.trellis_props
        index = props.multi_images_index

        if len(props.multi_images) == 0:
            self.report({'WARNING'}, "No images to remove")
            return {'CANCELLED'}

        if index < 0 or index >= len(props.multi_images):
            self.report({'WARNING'}, "Invalid selection")
            return {'CANCELLED'}

        # Remove image
        props.multi_images.remove(index)

        # Adjust index
        if index > 0:
            props.multi_images_index = index - 1
        else:
            props.multi_images_index = 0

        self.report({'INFO'}, "Image removed")
        return {'FINISHED'}


class TRELLIS_OT_ClearMultiImages(Operator):
    """Clear all images from multi-image list"""
    bl_idname = "trellis.clear_multi_images"
    bl_label = "Clear All"
    bl_description = "Remove all images from the list"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.trellis_props

        if len(props.multi_images) == 0:
            self.report({'INFO'}, "List is already empty")
            return {'CANCELLED'}

        # Clear all
        props.multi_images.clear()
        props.multi_images_index = 0

        self.report({'INFO'}, "All images cleared")
        return {'FINISHED'}


class TRELLIS_OT_GenerateFromMultiImage(Operator):
    """Generate 3D from multiple images"""
    bl_idname = "trellis.generate_from_multi_image"
    bl_label = "Generate from Multiple Images"
    bl_description = "Generate 3D model from multiple input images (multi-view)"
    bl_options = {'REGISTER'}

    def execute(self, context):
        props = context.scene.trellis_props

        # Check if we have images
        if len(props.multi_images) == 0:
            self.report({'ERROR'}, "No images in the list. Add at least one image.")
            return {'CANCELLED'}

        # Collect valid image paths
        image_paths = []
        for item in props.multi_images:
            if item.filepath and os.path.exists(item.filepath):
                image_paths.append(item.filepath)
            else:
                self.report({'WARNING'}, f"Skipping invalid path: {item.filepath}")

        if len(image_paths) == 0:
            self.report({'ERROR'}, "No valid images found")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Generating from {len(image_paths)} images...")

        # Import in execute to avoid circular dependency
        from .pipeline_manager import get_pipeline_manager

        manager = get_pipeline_manager()
        
        # Auto-initialize if not initialized
        if not manager.initialized:
            self.report({'INFO'}, "Initializing TRELLIS (first-time setup)...")
            import bpy
            preferences = bpy.context.preferences.addons[__package__].preferences
            if not manager.initialize(use_cuda=preferences.use_cuda):
                self.report({'ERROR'}, "Failed to initialize TRELLIS")
                return {'CANCELLED'}
            self.report({'INFO'}, "TRELLIS initialized successfully!")

        # Prepare formats
        formats = []
        if props.generate_mesh:
            formats.append('mesh')
        if props.generate_gaussian:
            formats.append('gaussian')

        if not formats:
            self.report({'ERROR'}, "No output format selected. Enable Mesh or Gaussian.")
            return {'CANCELLED'}

        try:
            # Generate from multiple images
            outputs = manager.generate_from_multi_image(
                image_paths=image_paths,
                seed=props.seed,
                sparse_steps=props.sparse_steps,
                sparse_cfg=props.sparse_cfg,
                slat_steps=props.slat_steps,
                slat_cfg=props.slat_cfg,
                preprocess=props.preprocess_image,
                formats=formats,
            )

            if outputs is None:
                self.report({'ERROR'}, "Generation failed")
                return {'CANCELLED'}

            # Export to GLB if mesh was generated
            if 'mesh' in outputs:
                from .mesh_utils import export_glb, import_glb, center_object, scale_object_to_size

                output_dir = bpy.path.abspath("//") or os.path.expanduser("~")
                output_path = os.path.join(output_dir, "trellis_multi_output.glb")

                try:
                    print(f"Exporting mesh to: {output_path}")
                    export_glb(
                        outputs['mesh'][0],
                        output_path,
                        texture_size=int(props.texture_size),
                        simplify=props.simplify_mesh,
                    )
                    print(f"✓ GLB exported successfully")

                    # Import into Blender
                    obj = import_glb(output_path)
                    if obj:
                        # Post-processing
                        if props.center_object:
                            center_object(obj)
                        if props.scale_object:
                            scale_object_to_size(obj, target_size=2.0)

                        # Make visible and select
                        obj.hide_viewport = False
                        obj.hide_render = False
                        obj.select_set(True)
                        context.view_layer.objects.active = obj

                        # Frame in viewport
                        for area in context.screen.areas:
                            if area.type == 'VIEW_3D':
                                for region in area.regions:
                                    if region.type == 'WINDOW':
                                        override = {'area': area, 'region': region}
                                        with context.temp_override(**override):
                                            bpy.ops.view3d.view_selected()
                                        break

                        self.report({'INFO'}, f"✓ Generated from {len(image_paths)} images! Object imported.")
                    else:
                        self.report({'WARNING'}, "Generated but import failed")

                except Exception as e:
                    self.report({'ERROR'}, f"Export failed: {e}")
                    return {'CANCELLED'}

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class TRELLIS_OT_GenerateMultiImageConsole(Operator):
    """Generate 3D from multiple images in separate console"""
    bl_idname = "trellis.generate_multi_image_console"
    bl_label = "Generate in Console"
    bl_description = "Run multi-image generation in separate console window with full logging"
    bl_options = {'REGISTER'}

    def execute(self, context):
        """Execute the operator"""
        import subprocess
        import platform
        import sys
        import json

        props = context.scene.trellis_props

        # Check if we have images
        if len(props.multi_images) == 0:
            self.report({'ERROR'}, "No images in the list. Add at least one image.")
            return {'CANCELLED'}

        # Collect valid image paths
        image_paths = []
        for item in props.multi_images:
            if item.filepath and os.path.exists(item.filepath):
                image_paths.append(item.filepath)

        if len(image_paths) == 0:
            self.report({'ERROR'}, "No valid images found")
            return {'CANCELLED'}

        # Get script path
        addon_dir = os.path.dirname(__file__)
        console_script = os.path.join(addon_dir, "generate_in_console.py")

        if not os.path.exists(console_script):
            self.report({'ERROR'}, f"Console script not found: {console_script}")
            return {'CANCELLED'}

        # Build command with all parameters from props
        generation_params = [
            '--seed', str(props.seed),
            '--sparse-steps', str(props.sparse_steps),
            '--sparse-cfg', str(props.sparse_cfg),
            '--slat-steps', str(props.slat_steps),
            '--slat-cfg', str(props.slat_cfg),
            '--texture-size', str(props.texture_size),
            '--mesh-simplify', str(props.simplify_mesh),
            '--multi-image',  # Flag for multi-image mode
        ]

        if props.preprocess_image:
            generation_params.append('--preprocess')

        print(f"\n{'='*70}")
        print("🚀 Launching multi-image generation in separate console...")
        print(f"   Images: {len(image_paths)} files")
        for i, path in enumerate(image_paths, 1):
            print(f"      {i}. {os.path.basename(path)}")
        print(f"   Script: {console_script}")
        print(f"   Parameters: seed={props.seed}, sparse_steps={props.sparse_steps}, slat_steps={props.slat_steps}")
        print(f"   Texture: {props.texture_size}x{props.texture_size}, Simplify: {props.simplify_mesh}")
        print(f"   Preprocess: {props.preprocess_image}")
        print(f"{'='*70}\n")

        try:
            if platform.system() == 'Windows':
                # Windows: Build command for multiple images
                cmd = [
                    'cmd.exe', '/k',
                    sys.executable,
                    console_script,
                ] + image_paths + generation_params
                
                print(f"   Full command: {' '.join(cmd)}\n")
                
                # Use creationflags to open new console window
                process = subprocess.Popen(
                    cmd,
                    cwd=addon_dir,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                
                self.report({'INFO'}, f"Console launched! Generating from {len(image_paths)} images...")
                print(f"✓ Console process started (PID: {process.pid})")
                print(f"  Check the console window for generation progress\n")
                
            else:
                # Unix/Linux/Mac
                cmd = [
                    sys.executable,
                    console_script,
                ] + image_paths + generation_params
                
                if platform.system() == 'Darwin':  # macOS
                    # macOS Terminal
                    applescript = f'''
                    tell application "Terminal"
                        do script "cd {addon_dir} && {' '.join(cmd)}"
                        activate
                    end tell
                    '''
                    subprocess.Popen(['osascript', '-e', applescript])
                else:  # Linux
                    # Try common terminal emulators
                    terminals = ['gnome-terminal', 'konsole', 'xterm']
                    for term in terminals:
                        try:
                            if term == 'gnome-terminal':
                                subprocess.Popen([term, '--', *cmd], cwd=addon_dir)
                            else:
                                subprocess.Popen([term, '-e', ' '.join(cmd)], cwd=addon_dir)
                            break
                        except FileNotFoundError:
                            continue
                
                self.report({'INFO'}, f"Console launched! Generating from {len(image_paths)} images...")

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Failed to launch console: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
