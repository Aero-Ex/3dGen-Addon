"""
UI panels for TRELLIS addon
"""

import bpy
from bpy.types import Panel
import time
import os

from .pipeline_metadata import get_pipeline_status_info, is_mode_supported

# Cache for installation status - only refreshed when explicitly requested
_status_cache = None
_status_initialized = False


def get_cached_status():
    """
    Get installation status with permanent caching

    Status is only checked:
    1. On first access (addon startup)
    2. When explicitly invalidated by operations (Install Dependencies, etc.)

    No periodic refreshes - status only changes when WE change it!
    """
    global _status_cache, _status_initialized

    # Check once on first access
    if not _status_initialized:
        _status_cache = _refresh_status()
        _status_initialized = True

    return _status_cache


def _refresh_status():
    """Actually perform the status check"""
    from . import dependency_installer
    try:
        # Use detailed=False for fast check (only venv + PyTorch, no 20+ subprocess calls!)
        return dependency_installer.get_installation_status(detailed=False)
    except Exception:
        # If status check fails, return minimal status
        return {
            'venv': {'exists': False, 'path': str(dependency_installer.get_venv_path())},
            'torch': {'installed': False, 'cuda': False, 'version': ''},
            'trellis': {'installed': False}
        }


def invalidate_status_cache():
    """Force status to be rechecked on next access"""
    global _status_cache, _status_initialized
    _status_initialized = False
    _status_cache = None


class TRELLIS_PT_MainPanel(Panel):
    """Main panel for TRELLIS 3D Generator"""
    bl_label = "TRELLIS 3D Generator"
    bl_idname = "TRELLIS_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRELLIS'

    def draw(self, context):
        layout = self.layout
        props = context.scene.trellis_props

        pipeline_info = get_pipeline_status_info(props.pipeline_engine)
        
        # Pipeline Selection
        row = layout.row(align=True)
        row.prop(props, "pipeline_engine", text="")
        
        # Status Icon & Hint
        row = layout.row(align=True)
        if pipeline_info['supported']:
            row.label(text=pipeline_info['status_hint'], icon=pipeline_info['status_icon'])
        else:
            row.alert = True
            row.label(text="Pipeline Unavailable", icon='ERROR')

        # Preferences Button
        layout.separator()
        layout.operator("trellis.open_preferences", icon='PREFERENCES', text="Preferences")


class TRELLIS_PT_SetupPanel(Panel):
    """Installation and Setup Panel"""
    bl_label = "Installation & Setup"
    bl_idname = "TRELLIS_PT_setup_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRELLIS'
    bl_parent_id = "TRELLIS_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        
        # Get cached installation status
        status = get_cached_status()
        venv_exists = status.get('venv', {}).get('exists', False)
        venv_path = status.get('venv', {}).get('path', '')
        torch_installed = status.get('torch', {}).get('installed', False)
        cuda_available = status.get('torch', {}).get('cuda', False)
        trellis_installed = status.get('trellis', {}).get('installed', False)
        installation_failed = status.get('installation_failed', False)

        # Check for DLL errors
        dll_error = False
        try:
            from .pipeline_manager import get_pipeline_manager
            manager = get_pipeline_manager()
        except Exception as e:
            if 'DLL' in str(e) or 'c10.dll' in str(e):
                dll_error = True

        # Venv Status
        box = layout.box()
        if venv_exists:
            row = box.row()
            row.label(text="Virtual Environment", icon='CHECKMARK')
            import os
            venv_folder = os.path.basename(venv_path)
            box.label(text=f"Path: .../{venv_folder}", icon='FILE_FOLDER')
        else:
            box.alert = True
            box.label(text="Virtual Environment Missing", icon='ERROR')

        # Dependency Status
        box = layout.box()
        if installation_failed:
            box.alert = True
            box.label(text="Installation Failed!", icon='ERROR')
            box.operator("trellis.install_dependencies", icon='FILE_REFRESH', text="Retry Installation")
        elif not venv_exists or not torch_installed:
            box.alert = True
            box.label(text="Dependencies Missing", icon='INFO')
            op = box.operator("trellis.install_dependencies", icon='IMPORT', text="Install Dependencies")
        elif not trellis_installed:
            box.alert = True
            box.label(text="TRELLIS Missing", icon='INFO')
            box.operator("trellis.install_dependencies", icon='IMPORT', text="Install TRELLIS")
        else:
            col = box.column(align=True)
            col.label(text=f"PyTorch {status['torch'].get('version', '')}", icon='CHECKMARK')
            col.label(text="CUDA Available" if cuda_available else "CUDA Unavailable", icon='CHECKMARK' if cuda_available else 'ERROR')
            col.label(text="TRELLIS Installed", icon='CHECKMARK')

        if dll_error:
            box = layout.box()
            box.alert = True
            box.label(text="DLL Error Detected!", icon='ERROR')
            box.label(text="Please reinstall dependencies.")


class TRELLIS_PT_InputPanel(Panel):
    """Input Selection Panel"""
    bl_label = "Input"
    bl_idname = "TRELLIS_PT_input_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRELLIS'
    bl_parent_id = "TRELLIS_PT_main_panel"

    def draw(self, context):
        layout = self.layout
        props = context.scene.trellis_props
        
        pipeline_info = get_pipeline_status_info(props.pipeline_engine)
        pipeline_supported = pipeline_info['supported']
        mode_supported = pipeline_supported and is_mode_supported(props.pipeline_engine, props.generation_mode)

        # Mode Selection
        layout.prop(props, "generation_mode", expand=True)
        layout.separator()

        if not mode_supported:
            layout.alert = True
            layout.label(text="Mode not supported by current pipeline", icon='ERROR')
            return

        # Image Mode
        if props.generation_mode == 'IMAGE':
            layout.prop(props, "input_image", text="")
            
            # Preview
            if props.input_image and os.path.exists(props.input_image):
                try:
                    image_name = os.path.basename(props.input_image)
                    img = bpy.data.images.get(image_name)
                    if img is None:
                        img = bpy.data.images.load(props.input_image, check_existing=True)
                    
                    if img:
                        img.preview_ensure()
                        if img.preview:
                            box = layout.box()
                            col = box.column()
                            col.template_icon(icon_value=img.preview.icon_id, scale=5.0)
                except:
                    pass
            
            layout.prop(props, "preprocess_image", text="Remove Background")
            
            layout.separator()
            row = layout.row()
            row.scale_y = 1.5
            row.operator("trellis.generate_image_console", icon='PLAY', text="Generate 3D")

        # Multi-Image Mode
        elif props.generation_mode == 'MULTI_IMAGE':
            row = layout.row()
            row.template_list("TRELLIS_UL_ImageList", "", props, "multi_images", props, "multi_images_index", rows=3)
            
            col = row.column(align=True)
            col.operator("trellis.add_multi_image", icon='ADD', text="")
            col.operator("trellis.remove_multi_image", icon='REMOVE', text="")
            col.operator("trellis.clear_multi_images", icon='X', text="")
            
            layout.prop(props, "preprocess_image", text="Remove Background")
            
            layout.separator()
            row = layout.row()
            row.scale_y = 1.5
            row.operator("trellis.generate_multi_image_console", icon='PLAY', text="Generate 3D")

        # Text Mode
        elif props.generation_mode == 'TEXT':
            layout.prop(props, "text_prompt", text="")
            
            layout.separator()
            row = layout.row()
            row.scale_y = 1.5
            row.operator("trellis.generate_text_console", icon='PLAY', text="Generate 3D")

        # Upscale Mode
        elif props.generation_mode == 'UPSCALE':
            layout.label(text="Reference Image:", icon='IMAGE_DATA')
            layout.prop(props, "input_image", text="")
            
            # Preview
            if props.input_image and os.path.exists(props.input_image):
                try:
                    image_name = os.path.basename(props.input_image)
                    img = bpy.data.images.get(image_name)
                    if img is None:
                        img = bpy.data.images.load(props.input_image, check_existing=True)
                    
                    if img:
                        img.preview_ensure()
                        if img.preview:
                            box = layout.box()
                            col = box.column()
                            col.template_icon(icon_value=img.preview.icon_id, scale=5.0)
                except:
                    pass
            
            layout.separator()
            layout.label(text="Mesh to Upscale:", icon='MESH_DATA')
            layout.prop(props, "input_mesh", text="")

            layout.separator()
            layout.prop(props, "upscale_resolution", text="Resolution")

            layout.separator()
            row = layout.row()
            row.scale_y = 1.5
            row.operator("trellis.upscale_direct3d_console", icon='MOD_SUBSURF', text="Upscale Mesh")


class TRELLIS_PT_SettingsPanel(Panel):
    """Generation Settings Panel"""
    bl_label = "Generation Settings"
    bl_idname = "TRELLIS_PT_settings_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRELLIS'
    bl_parent_id = "TRELLIS_PT_main_panel"

    def draw(self, context):
        layout = self.layout
        props = context.scene.trellis_props
        
        layout.prop(props, "seed")
        
        if props.pipeline_engine == 'TRELLIS':
            box = layout.box()
            box.label(text="Structure (Stage 1)", icon='MESH_CUBE')
            col = box.column(align=True)
            col.prop(props, "sparse_steps", text="Steps")
            col.prop(props, "sparse_cfg", text="CFG")
            
            box = layout.box()
            box.label(text="Appearance (Stage 2)", icon='MATERIAL')
            col = box.column(align=True)
            col.prop(props, "slat_steps", text="Steps")
            col.prop(props, "slat_cfg", text="CFG")
            

        elif props.pipeline_engine == 'DIRECT3D':
            box = layout.box()
            box.label(text="Direct3D-S2 Settings", icon='SURFACE_DATA')
            col = box.column(align=True)
            col.prop(props, "direct3d_steps", text="Steps")
            col.prop(props, "direct3d_guidance", text="Guidance")
            col.prop(props, "direct3d_resolution", text="Resolution")

            # Mesh simplification settings
            col.separator()
            col.prop(props, "enable_simplify_mesh", text="Simplify Mesh")
            if props.enable_simplify_mesh:
                col.prop(props, "simplify_mesh", text="Simplify Ratio")


class TRELLIS_PT_OutputPanel(Panel):
    """Output Settings Panel"""
    bl_label = "Output"
    bl_idname = "TRELLIS_PT_output_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRELLIS'
    bl_parent_id = "TRELLIS_PT_main_panel"

    def draw(self, context):
        layout = self.layout
        props = context.scene.trellis_props
        
        col = layout.column(align=True)
        
        # Output formats based on pipeline
        if props.pipeline_engine == 'TRELLIS':
            col.prop(props, "generate_mesh")
            col.prop(props, "generate_gaussian")
        elif props.pipeline_engine == 'DIRECT3D':
            # Direct3D only supports mesh
            row = col.row()
            row.enabled = False
            row.prop(props, "generate_mesh")

        
        layout.separator()

        # Pipeline-specific output settings
        if props.pipeline_engine == 'TRELLIS':
            # TRELLIS has texture baking options
            col = layout.column(align=True)
            col.prop(props, "export_mode", text="Export")

            if props.export_mode == 'MESH_TEXTURE':
                col.prop(props, "texture_size")

            col.prop(props, "simplify_mesh", slider=True)

        elif props.pipeline_engine == 'DIRECT3D':
            # Direct3D simplification is controlled in Settings panel
            box = layout.box()
            box.label(text="Direct3D outputs raw mesh", icon='INFO')
            box.label(text="Use Settings panel for mesh simplification")



        layout.separator()

        # Post-processing options (common to all)
        col = layout.column(align=True)
        col.prop(props, "center_object")
        col.prop(props, "scale_object")
        col.prop(props, "convert_to_quads")
        
        layout.separator()
        layout.operator("trellis.import_last_generation", icon='IMPORT', text="Import Last Result")


class TRELLIS_PT_ProgressPanel(Panel):
    """Progress Panel"""
    bl_label = "Progress"
    bl_idname = "TRELLIS_PT_progress_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRELLIS'
    bl_parent_id = "TRELLIS_PT_main_panel"

    @classmethod
    def poll(cls, context):
        from .pipeline_manager import get_generation_progress
        return get_generation_progress()['active']

    def draw(self, context):
        layout = self.layout
        from .pipeline_manager import get_generation_progress
        progress = get_generation_progress()
        
        col = layout.column(align=True)
        col.label(text=progress['stage'], icon='TIME')
        
        if progress['total_steps'] > 0:
            pct = (progress['step'] / progress['total_steps']) * 100
            col.label(text=f"{progress['step']}/{progress['total_steps']} ({pct:.0f}%)")
            
        if progress['message']:
            col.label(text=progress['message'], icon='INFO')
