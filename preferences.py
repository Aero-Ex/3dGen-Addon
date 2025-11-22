"""
Addon preferences for TRELLIS 3D Generator
"""

import bpy
from bpy.props import StringProperty, BoolProperty, IntProperty, FloatProperty, EnumProperty
from bpy.types import AddonPreferences


class TRELLIS_AddonPreferences(AddonPreferences):
    """Preferences for TRELLIS addon"""
    bl_idname = __package__

    cache_models: BoolProperty(
        name="Cache Models",
        description="Keep models loaded in memory (faster but uses more RAM)",
        default=True,
    )

    use_cuda: BoolProperty(
        name="Use CUDA (GPU)",
        description="Use GPU acceleration. Requires NVIDIA GPU with 16GB+ VRAM (RTX 3090/4090, A6000, etc.)",
        default=True,
    )

    enable_smart_offload: BoolProperty(
        name="Enable Smart Model Offloading",
        description=(
            "Automatically offload models between GPU and CPU RAM for low VRAM systems (≤8GB). "
            "Slower but prevents OOM crashes. Auto-detects VRAM and enables when needed. "
            "⚠ May cause device errors with text-to-3D on some systems."
        ),
        default=True,
    )




    mesh_simplify: FloatProperty(
        name="Mesh Simplification",
        description="Mesh simplification ratio (higher = more simplified, smaller file)",
        default=0.95,
        min=0.5,
        max=0.99,
    )

    def draw(self, context):
        layout = self.layout

        # Runtime Settings
        box = layout.box()
        box.label(text="Runtime Settings:", icon='PREFERENCES')
        box.prop(self, "use_cuda")
        box.prop(self, "enable_smart_offload")
        box.prop(self, "cache_models")

        # Show info about offloading
        if self.enable_smart_offload:
            info_box = layout.box()
            info_box.label(text="Smart Offloading Info:", icon='INFO')
            info_box.label(text="• Auto-detects GPU VRAM (≤8GB → offloading ON)")
            info_box.label(text="• Moves models between GPU/CPU as needed")
            info_box.label(text="• Works with 6GB VRAM + 16GB RAM")
            info_box.label(text="• Slower but prevents crashes (5-15 min per model)")

       

        # Requirements
        box = layout.box()
        box.label(text="Requirements:", icon='ERROR')
        box.label(text="• Python 3.10+ (Blender's Python)")
        if self.enable_smart_offload:
            box.label(text="• NVIDIA GPU with 6GB+ VRAM (with offloading)")
            box.label(text="• 16GB+ System RAM recommended")
        else:
            box.label(text="• NVIDIA GPU with 16GB+ VRAM (standard mode)")
        box.label(text="• PyTorch with CUDA support")
        box.label(text="• 10-30 minutes for first-time dependency installation")
