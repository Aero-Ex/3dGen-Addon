"""
UI panels for TRELLIS addon
"""

import bpy
from bpy.types import Panel
import time
import os

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

        # Get cached installation status (checked once on startup, updated only when we change it)
        status = get_cached_status()
        venv_exists = status.get('venv', {}).get('exists', False)
        venv_path = status.get('venv', {}).get('path', '')
        torch_installed = status.get('torch', {}).get('installed', False)
        cuda_available = status.get('torch', {}).get('cuda', False)
        trellis_installed = status.get('trellis', {}).get('installed', False)
        installation_failed = status.get('installation_failed', False)

        # Get pipeline manager status (wrapped in try/except to handle DLL errors)
        manager_initialized = False
        dll_error = False
        try:
            from .pipeline_manager import get_pipeline_manager
            manager = get_pipeline_manager()
            manager_initialized = manager.initialized
        except Exception as e:
            # DLL errors or import failures are handled gracefully
            error_str = str(e)
            if 'DLL' in error_str or 'c10.dll' in error_str:
                dll_error = True

        # Preferences access button
        row = layout.row()
        row.scale_y = 1.2
        row.operator("trellis.open_preferences", icon='PREFERENCES', text="Addon Preferences (Smart Offloading)")

        layout.separator()

        # Initialization section
        box = layout.box()
        box.label(text="Setup:", icon='SETTINGS')

        # Show venv status
        if venv_exists:
            col = box.column(align=True)
            col.label(text="✓ Virtual environment ready", icon='CHECKMARK')
            # Show venv path in a compact way
            import os
            venv_folder = os.path.basename(venv_path)
            venv_parent = os.path.basename(os.path.dirname(venv_path))
            col.label(text=f"   {venv_parent}/{venv_folder}", icon='FILE_FOLDER')
            box.separator()

        # Dependency status
        if installation_failed:
            # Installation was attempted but failed
            box.label(text="⚠ Installation Failed!", icon='ERROR')
            row = box.row()
            row.scale_y = 1.3
            row.operator("trellis.install_dependencies", icon='FILE_REFRESH', text="Try Again")
            box.label(text="Click 'Try Again' to reinstall", icon='INFO')
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
            # Show status
            col = box.column(align=True)
            col.label(text=f"✓ PyTorch {status['torch'].get('version', '')}", icon='CHECKMARK')
            if cuda_available:
                col.label(text="✓ CUDA available", icon='CHECKMARK')
            else:
                col.label(text="⚠ CUDA not available", icon='ERROR')
            col.label(text="✓ TRELLIS installed", icon='CHECKMARK')

        box.separator()

        # Show DLL error warning if detected
        if dll_error:
            box.label(text="⚠ PyTorch DLL Error Detected!", icon='ERROR')
            box.label(text="Re-run 'Install Dependencies'", icon='INFO')
            box.label(text="to fix PyTorch installation", icon='INFO')
            box.separator()

        # Pipeline status (auto-initializes on first generation)
        if not manager_initialized:
            if torch_installed and trellis_installed and not dll_error:
                box.label(text="Ready to generate", icon='INFO')
                # box.label(text="(auto-initializes on first use)", icon='BLANK1')
            else:
                if dll_error:
                    box.label(text="Fix DLL error first", icon='CANCEL')
                else:
                    box.label(text="Install dependencies first", icon='CANCEL')
        else:
            box.label(text="Pipeline ready", icon='CHECKMARK')

        layout.separator()

        # Advanced Settings (Generation Parameters)
        box = layout.box()
        box.label(text="Generation Parameters:", icon='SETTINGS')
        
        # Seed setting
        box.prop(props, "seed", text="Random Seed")

        # Sparse structure settings
        col = box.column(align=True)
        col.label(text="Sparse Structure (Stage 1):")
        col.prop(props, "sparse_steps", text="Steps")
        col.prop(props, "sparse_cfg", text="CFG Strength")

        box.separator()

        # SLAT settings
        col = box.column(align=True)
        col.label(text="SLAT Appearance (Stage 2):")
        col.prop(props, "slat_steps", text="Steps")
        col.prop(props, "slat_cfg", text="CFG Strength")

        layout.separator()

        # Progress indicator
        from .pipeline_manager import get_generation_progress
        progress = get_generation_progress()

        if progress['active']:
            box = layout.box()
            box.label(text="Generation Progress:", icon='TIME')

            col = box.column(align=True)
            col.label(text=f"Stage: {progress['stage']}", icon='SETTINGS')

            if progress['total_steps'] > 0:
                percentage = (progress['step'] / progress['total_steps']) * 100
                col.label(text=f"Step: {progress['step']}/{progress['total_steps']} ({percentage:.1f}%)")

            if progress['message']:
                col.label(text=f"{progress['message']}", icon='INFO')

            layout.separator()

        # ====================
        # Navigation Tabs
        # ====================
        box = layout.box()
        row = box.row(align=True)
        row.prop(props, "generation_mode", expand=True)
        
        layout.separator()

        # ====================
        # Image to 3D Mode
        # ====================
        if props.generation_mode == 'IMAGE':
            box = layout.box()
            box.label(text="Image to 3D:", icon='IMAGE_DATA')

            box.prop(props, "input_image")
            
            # Show image preview if a valid image is selected
            if props.input_image and os.path.exists(props.input_image):
                try:
                    # Load image into Blender if not already loaded
                    image_name = os.path.basename(props.input_image)
                    img = bpy.data.images.get(image_name)
                    
                    # If not found, load it
                    if img is None:
                        img = bpy.data.images.load(props.input_image, check_existing=True)
                    
                    if img and img.size[0] > 0:
                        # Ensure preview is generated
                        img.preview_ensure()
                        
                        # Display image preview in a box
                        preview_box = box.box()
                        preview_box.label(text="Image Preview:", icon='IMAGE_DATA')
                        
                        # Draw the image preview using template_icon with proper icon_value
                        if img.preview and img.preview.icon_id > 0:
                            col = preview_box.column(align=True)
                            col.scale_y = 3.0  # Moderate size - not too big, not too small
                            col.template_icon(icon_value=img.preview.icon_id, scale=5.0)
                        
                        # Show image info
                        info_row = preview_box.row()
                        info_row.label(text=f"Size: {img.size[0]} x {img.size[1]} px", icon='INFO')
                except Exception as e:
                    # If preview fails, just show a simple message
                    info_box = box.box()
                    info_box.label(text=f"Image: {os.path.basename(props.input_image)}", icon='IMAGE_DATA')
            
            box.prop(props, "preprocess_image")

            # Console generation button
            row = box.row()
            row.scale_y = 1.5
            row.operator("trellis.generate_image_console", icon='CONSOLE', text="Generate from Image")
            row.enabled = torch_installed and trellis_installed and not dll_error

            layout.separator()

        # ====================
        # Multi-Image to 3D Mode
        # ====================
        elif props.generation_mode == 'MULTI_IMAGE':
            box = layout.box()
            box.label(text="Multi-Image to 3D (Multi-View):", icon='IMAGE')
            
            box.label(text="Select multiple images from different angles:", icon='INFO')

            # UI List for multiple images
            row = box.row()
            row.template_list(
                "TRELLIS_UL_ImageList", "",
                props, "multi_images",
                props, "multi_images_index",
                rows=4
            )

            # List controls
            col = row.column(align=True)
            col.operator("trellis.add_multi_image", icon='ADD', text="")
            col.operator("trellis.remove_multi_image", icon='REMOVE', text="")
            col.separator()
            col.operator("trellis.clear_multi_images", icon='X', text="")

            # Show count
            count = len(props.multi_images)
            box.label(text=f"Images: {count}", icon='RENDERLAYERS')
            
            # Preprocess option
            box.prop(props, "preprocess_image", text="Remove Background")

            # Generate button
            row = box.row()
            row.scale_y = 1.5
            row.operator("trellis.generate_multi_image_console", icon='CONSOLE', text="Generate from Multiple Images")
            row.enabled = torch_installed and trellis_installed and not dll_error and count > 0

            layout.separator()

        # ====================
        # Text to 3D Mode
        # ====================
        elif props.generation_mode == 'TEXT':
            box = layout.box()
            box.label(text="Text to 3D:", icon='TEXT')

            box.prop(props, "text_prompt", text="")

            # Console generation button
            row = box.row()
            row.scale_y = 1.5
            row.operator("trellis.generate_text_console", icon='CONSOLE', text="Generate from Text")
            row.enabled = torch_installed and trellis_installed and not dll_error

            layout.separator()

        # ====================
        # Import Last Generation (Common)
        # ====================
        box = layout.box()
        row = box.row()
        row.scale_y = 1.2
        row.operator("trellis.import_last_generation", icon='IMPORT', text="Import Last Generation")

        layout.separator()

        # Output settings
        box = layout.box()
        box.label(text="Output Settings:", icon='EXPORT')

        row = box.row()
        row.prop(props, "generate_mesh")
        row.prop(props, "generate_gaussian")

        # Export quality settings
        box.prop(props, "texture_size")
        box.prop(props, "simplify_mesh", slider=True)

        box.separator()

        box.prop(props, "center_object")
        box.prop(props, "scale_object")
