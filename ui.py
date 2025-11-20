"""
UI panels for TRELLIS addon
"""

import bpy
from bpy.types import Panel
import os
import threading

# Cache for installation status
_status_cache = None
_status_initialized = False
_status_checking = False
_check_status_thread = None


def _check_status_thread():
    """Background thread to check status"""
    global _status_cache, _status_checking, _status_initialized
    
    _status_cache = _refresh_status()
    _status_initialized = True
    _status_checking = False
    
    # Redraw UI to show results
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

def get_cached_status():
    """
    Get installation status with non-blocking background check
    """
    global _status_cache, _status_checking, _status_initialized

    # If we have a result, return it
    if _status_initialized and _status_cache is not None:
        return _status_cache

    # If not initialized and not checking, start check
    if not _status_initialized and not _status_checking:
        _status_checking = True
        thread = threading.Thread(target=_check_status_thread)
        thread.daemon = True
        thread.start()
        
    # Return loading state
    return {'loading': True}


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
    global _status_cache, _status_initialized, _status_checking
    _status_initialized = False
    _status_cache = None
    _status_checking = False


class TRELLIS_PT_MainPanel(Panel):
    """Main panel for TRELLIS 3D Generator"""
    bl_label = "TRELLIS 3D Generator"
    bl_idname = "TRELLIS_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'TRELLIS'

    def draw(self, context):
        layout = self.layout
        # Main panel is now just a container for sub-panels
        # We can put global status or important messages here if needed
        pass

class TRELLIS_PT_SetupPanel(Panel):
    """Setup and Installation Panel"""
    bl_label = "Setup & Installation"
    bl_idname = "TRELLIS_PT_setup_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_parent_id = "TRELLIS_PT_main_panel"

    def draw(self, context):
        layout = self.layout
        props = context.scene.trellis_props

        # Get cached installation status
        status = get_cached_status()
        
        # Handle loading state
        if status.get('loading', False):
            box = layout.box()
            box.label(text="Checking dependencies...", icon='TIME')
            return

        venv_exists = status.get('venv', {}).get('exists', False)
        venv_path = status.get('venv', {}).get('path', '')
        torch_installed = status.get('torch', {}).get('installed', False)
        cuda_available = status.get('torch', {}).get('cuda', False)
        trellis_installed = status.get('trellis', {}).get('installed', False)
        installation_failed = status.get('installation_failed', False)

        # Get pipeline manager status
        manager_initialized = False
        dll_error = False
        try:
            from .pipeline_manager import get_pipeline_manager
            manager = get_pipeline_manager()
            manager_initialized = manager.initialized
        except Exception as e:
            error_str = str(e)
            if 'DLL' in error_str or 'c10.dll' in error_str:
                dll_error = True

        # Preferences access button
        row = layout.row()
        row.operator("trellis.open_preferences", icon='PREFERENCES', text="Preferences")

        layout.separator()

        # Initialization section
        box = layout.box()
        box.label(text="Status:", icon='INFO')

        # Show venv status
        if venv_exists:
            col = box.column(align=True)
            col.label(text="✓ Virtual environment ready", icon='CHECKMARK')
            import os
            venv_folder = os.path.basename(venv_path)
            venv_parent = os.path.basename(os.path.dirname(venv_path))
            col.label(text=f"   {venv_parent}/{venv_folder}", icon='FILE_FOLDER')
            box.separator()

        # Dependency status
        if installation_failed:
            box.label(text="⚠ Installation Failed!", icon='ERROR')
            row = box.row()
            row.scale_y = 1.3
            row.operator("trellis.install_dependencies", icon='FILE_REFRESH', text="Try Again")
        elif not venv_exists or not torch_installed:
            box.label(text="Dependencies not installed", icon='ERROR')
            row = box.row()
            row.scale_y = 1.3
            row.operator("trellis.install_dependencies", icon='IMPORT')
            box.label(text="⚠ This will take 10-30 minutes", icon='TIME')
        elif not trellis_installed:
            box.label(text="PyTorch installed, TRELLIS missing", icon='ERROR')
            box.operator("trellis.install_dependencies", icon='IMPORT')
        else:
            col = box.column(align=True)
            col.label(text=f"✓ PyTorch {status['torch'].get('version', '')}", icon='CHECKMARK')
            if cuda_available:
                col.label(text="✓ CUDA available", icon='CHECKMARK')
            else:
                col.label(text="⚠ CUDA not available", icon='ERROR')
            col.label(text="✓ TRELLIS installed", icon='CHECKMARK')

        box.separator()

        # Show DLL error warning
        if dll_error:
            box.label(text="⚠ PyTorch DLL Error Detected!", icon='ERROR')
            box.label(text="Re-run 'Install Dependencies'", icon='INFO')
            box.separator()

        # Pipeline status
        if not manager_initialized:
            if torch_installed and trellis_installed and not dll_error:
                box.label(text="Ready to generate", icon='INFO')
            else:
                if dll_error:
                    box.label(text="Fix DLL error first", icon='CANCEL')
                else:
                    box.label(text="Install dependencies first", icon='CANCEL')
        else:
            box.label(text="Pipeline ready", icon='CHECKMARK')


class TRELLIS_PT_InputPanel(Panel):
    """Input Selection Panel"""
    bl_label = "Input"
    bl_idname = "TRELLIS_PT_input_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_parent_id = "TRELLIS_PT_main_panel"

    def draw(self, context):
        layout = self.layout
        props = context.scene.trellis_props
        
        # Get status for enabling/disabling buttons
        status = get_cached_status()
        torch_installed = status.get('torch', {}).get('installed', False)
        trellis_installed = status.get('trellis', {}).get('installed', False)
        
        # Check for DLL error
        dll_error = False
        try:
            from .pipeline_manager import get_pipeline_manager
            # Just check if import works, don't need manager instance here
        except Exception as e:
            if 'DLL' in str(e) or 'c10.dll' in str(e):
                dll_error = True

        # Mode Selection
        row = layout.row(align=True)
        row.prop(props, "generation_mode", expand=True)
        
        layout.separator()

        # Image to 3D Mode
        if props.generation_mode == 'IMAGE':
            col = layout.column(align=True)
            col.prop(props, "input_image")
            
            # Preview
            if props.input_image and os.path.exists(props.input_image):
                self._draw_image_preview(layout, props.input_image)
            
            layout.separator()
            layout.prop(props, "preprocess_image")

            # Generate Button
            row = layout.row()
            row.scale_y = 1.5
            row.operator("trellis.generate_image_console", icon='CONSOLE', text="Generate from Image")
            row.enabled = torch_installed and trellis_installed and not dll_error

        # Multi-Image Mode
        elif props.generation_mode == 'MULTI_IMAGE':
            layout.label(text="Select multiple images from different angles:", icon='INFO')

            # UI List
            row = layout.row()
            row.template_list(
                "TRELLIS_UL_ImageList", "",
                props, "multi_images",
                props, "multi_images_index",
                rows=4
            )

            # Controls
            col = row.column(align=True)
            col.operator("trellis.add_multi_image", icon='ADD', text="")
            col.operator("trellis.remove_multi_image", icon='REMOVE', text="")
            col.separator()
            col.operator("trellis.clear_multi_images", icon='X', text="")

            # Count
            count = len(props.multi_images)
            layout.label(text=f"Images: {count}", icon='RENDERLAYERS')
            
            layout.prop(props, "preprocess_image", text="Remove Background")

            # Generate Button
            row = layout.row()
            row.scale_y = 1.5
            row.operator("trellis.generate_multi_image_console", icon='CONSOLE', text="Generate from Multiple Images")
            row.enabled = torch_installed and trellis_installed and not dll_error and count > 0

        # Text Mode
        elif props.generation_mode == 'TEXT':
            layout.prop(props, "text_prompt", text="")

            # Generate Button
            row = layout.row()
            row.scale_y = 1.5
            row.operator("trellis.generate_text_console", icon='CONSOLE', text="Generate from Text")
            row.enabled = torch_installed and trellis_installed and not dll_error
            
        # Import Last Generation (Always available)
        layout.separator()
        row = layout.row()
        row.operator("trellis.import_last_generation", icon='IMPORT', text="Import Last Generation")

    def _draw_image_preview(self, layout, image_path):
        try:
            image_name = os.path.basename(image_path)
            img = bpy.data.images.get(image_name)
            if img is None:
                img = bpy.data.images.load(image_path, check_existing=True)
            
            if img and img.size[0] > 0:
                img.preview_ensure()
                box = layout.box()
                if img.preview and img.preview.icon_id > 0:
                    col = box.column(align=True)
                    col.template_icon(icon_value=img.preview.icon_id, scale=5.0)
                box.label(text=f"{img.size[0]} x {img.size[1]} px", icon='INFO')
        except:
            pass


class TRELLIS_PT_GenerationPanel(Panel):
    """Generation Parameters Panel"""
    bl_label = "Generation Settings"
    bl_idname = "TRELLIS_PT_generation_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_parent_id = "TRELLIS_PT_main_panel"

    def draw(self, context):
        layout = self.layout
        props = context.scene.trellis_props
        
        layout.prop(props, "seed", text="Random Seed")
        
        layout.separator()
        
        col = layout.column(align=True)
        col.label(text="Stage 1: Sparse Structure")
        col.prop(props, "sparse_steps", text="Steps")
        col.prop(props, "sparse_cfg", text="CFG Strength")

        layout.separator()

        col = layout.column(align=True)
        col.label(text="Stage 2: SLAT Appearance")
        col.prop(props, "slat_steps", text="Steps")
        col.prop(props, "slat_cfg", text="CFG Strength")
        
        layout.separator()
        layout.prop(props, "generate_texture")
        
        layout.separator()
        
        col = layout.column(align=True)
        col.label(text="Mesh Settings:")
        col.prop(props, "texture_size")
        col.prop(props, "simplify_mesh", slider=True)
        
        # Progress indicator
        from .pipeline_manager import get_generation_progress
        progress = get_generation_progress()

        if progress['active']:
            layout.separator()
            box = layout.box()
            box.label(text="Generation Progress:", icon='TIME')
            col = box.column(align=True)
            col.label(text=f"Stage: {progress['stage']}", icon='SETTINGS')
            if progress['total_steps'] > 0:
                percentage = (progress['step'] / progress['total_steps']) * 100
                col.label(text=f"Step: {progress['step']}/{progress['total_steps']} ({percentage:.1f}%)")
            if progress['message']:
                col.label(text=f"{progress['message']}", icon='INFO')


class TRELLIS_PT_OutputPanel(Panel):
    """Output Settings Panel"""
    bl_label = "Output Settings"
    bl_idname = "TRELLIS_PT_output_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_parent_id = "TRELLIS_PT_main_panel"

    def draw(self, context):
        layout = self.layout
        props = context.scene.trellis_props

        col = layout.column(align=True)
        col.label(text="Formats:")
        row = col.row(align=True)
        row.prop(props, "generate_mesh")
        row.prop(props, "generate_gaussian")

        layout.separator()

        col = layout.column(align=True)
        col.prop(props, "center_object")
        col.prop(props, "scale_object")
