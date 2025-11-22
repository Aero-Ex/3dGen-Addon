"""  
Blender operators for TRELLIS 3D generation
"""

import bpy
from bpy.props import (
    StringProperty,
    IntProperty,
    FloatProperty,
    BoolProperty,
    EnumProperty,
    CollectionProperty,
    PointerProperty,
)
from bpy.types import Operator, PropertyGroup, UIList
import os

from .pipeline_metadata import (
    PIPELINE_ENUM_ITEMS,
    ensure_pipeline_supported,
    get_pipeline_status_info,
    pipeline_cli_value,
)


def get_generation_modes(self, context):
    """Dynamic generation modes based on selected pipeline"""
    from .pipeline_metadata import _PIPELINES
    
    items = []
    # Default to TRELLIS if something goes wrong or during registration
    engine = 'TRELLIS'
    
    # Try to get current engine from self
    if context and hasattr(context.scene, 'trellis_props'):
        engine = context.scene.trellis_props.pipeline_engine
    elif hasattr(self, 'pipeline_engine'):
        engine = self.pipeline_engine
        
    if engine in _PIPELINES:
        modes = _PIPELINES[engine].modes
        
        if modes.get('IMAGE', False):
            items.append(('IMAGE', "Image to 3D", "Generate 3D from single image", 'IMAGE_DATA', 0))
            
        if modes.get('MULTI_IMAGE', False):
            items.append(('MULTI_IMAGE', "Multi-Image", "Generate 3D from multiple images", 'IMAGE', 1))
            
        if modes.get('TEXT', False):
            items.append(('TEXT', "Text to 3D", "Generate 3D from text prompt", 'TEXT', 2))
            
        if modes.get('UPSCALE', False):
            items.append(('UPSCALE', "Upscale Mesh", "Upscale existing mesh", 'MOD_SUBSURF', 3))
            
    # Fallback if no modes found (shouldn't happen)
    if not items:
        items.append(('IMAGE', "Image to 3D", "Generate 3D from single image", 'IMAGE_DATA', 0))
        
    return items


def update_pipeline_engine(self, context):
    """Update generation mode when pipeline changes"""
    # Get valid modes for new pipeline
    modes = get_generation_modes(self, context)
    valid_keys = [m[0] for m in modes]
    
    # If current mode is not valid, switch to first valid mode
    if self.generation_mode not in valid_keys:
        if valid_keys:
            self.generation_mode = valid_keys[0]
            
    # Set default output settings based on pipeline
    if self.pipeline_engine == 'DIRECT3D':
        self.generate_mesh = True
        self.generate_gaussian = False


class TRELLIS_ImageItem(PropertyGroup):
    """Single image item for multi-image generation"""
    filepath: StringProperty(
        name="Image Path",
        description="Path to image file",
        default="",
        subtype='FILE_PATH',
    )


class TRELLIS_Properties(PropertyGroup):
    """Properties for TRELLIS addon"""

    # Navigation mode
    generation_mode: EnumProperty(
        name="Generation Mode",
        description="Select generation mode",
        items=get_generation_modes,
        default=0,
    )

    pipeline_engine: EnumProperty(
        name="Pipeline",
        description="Select which backend pipeline to run",
        items=PIPELINE_ENUM_ITEMS,
        default='TRELLIS',
        update=update_pipeline_engine,
    )

    # Input settings
    input_image: StringProperty(
        name="Input Image",
        description="Path to input image for image-to-3D generation",
        default="",
        subtype='FILE_PATH',
    )

    input_mesh: PointerProperty(
        name="Input Mesh",
        description="Mesh to upscale",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH'
    )

    # Multi-image settings
    multi_images: CollectionProperty(
        type=TRELLIS_ImageItem,
        name="Multiple Images",
        description="Collection of images for multi-view generation",
    )

    multi_images_index: IntProperty(
        name="Active Image Index",
        description="Currently selected image in the list",
        default=0,
    )

    text_prompt: StringProperty(
        name="Text Prompt",
        description="Text prompt for text-to-3D generation",
        default="A chair looking like an avocado",
        maxlen=1024,
    )

    # Generation settings
    seed: IntProperty(
        name="Seed",
        description="Random seed for reproducible generation",
        default=42,
        min=0,
        max=999999,
    )

    # Sparse structure settings
    sparse_steps: IntProperty(
        name="Sparse Steps",
        description="Number of steps for sparse structure sampling (higher = better quality, slower)",
        default=8,
        min=1,
        max=50,
    )

    sparse_cfg: FloatProperty(
        name="Sparse CFG",
        description="Classifier-free guidance strength for sparse structure",
        default=7.5,
        min=0.0,
        max=20.0,
    )

    # SLAT settings
    slat_steps: IntProperty(
        name="SLAT Steps",
        description="Number of steps for SLAT sampling (higher = better quality, slower)",
        default=8,
        min=1,
        max=50,
    )

    slat_cfg: FloatProperty(
        name="SLAT CFG",
        description="Classifier-free guidance strength for SLAT",
        default=3.0,
        min=0.0,
        max=20.0,
    )

    # Processing settings
    preprocess_image: BoolProperty(
        name="Remove Background",
        description="Automatically remove background from input image",
        default=True,
    )

    generate_mesh: BoolProperty(
        name="Generate Mesh",
        description="Generate mesh output",
        default=True,
    )

    convert_to_quads: BoolProperty(
        name="Convert to Quads",
        description="Convert triangles to quads after generation",
        default=False,
    )

    generate_gaussian: BoolProperty(
        name="Generate Gaussian",
        description="Generate 3D Gaussian output",
        default=False,
    )

    # Output settings
    enable_simplify_mesh: BoolProperty(
        name="Enable Mesh Simplification",
        description="Simplify the generated mesh to reduce polygon count. Disable for maximum detail (warning: very high poly count, 2-5M polygons!)",
        default=True,
    )

    simplify_mesh: FloatProperty(
        name="Simplify Ratio",
        description="Mesh simplification ratio (1.0 = no simplification, 0.95 = 5% reduction)",
        default=0.95,
        min=0.1,
        max=1.0,
    )



    # Direct3D-S2 settings
    direct3d_steps: IntProperty(
        name="Inference Steps",
        description="Number of denoising steps (higher = better quality, slower)",
        default=15,
        min=1,
        max=100,
    )

    direct3d_guidance: FloatProperty(
        name="Guidance Scale",
        description="Classifier-free guidance scale (higher = closer to prompt/image)",
        default=7.0,
        min=1.0,
        max=20.0,
    )
    
    direct3d_resolution: EnumProperty(
        name="SDF Resolution",
        description="3D resolution for Direct3D-S2 generation (512 = 6GB+ VRAM, 1024 = 24GB+ VRAM)",
        items=[
            ('AUTO', 'Auto', 'Automatically select based on GPU memory (512 for <24GB, 1024 for >=24GB)'),
            ('512', '512³', 'Low resolution (faster, 6GB+ VRAM)'),
            ('1024', '1024³', 'High resolution (slower, 24GB+ VRAM)'),
        ],
        default='AUTO',
    )

    upscale_resolution: EnumProperty(
        name="Upscale Resolution",
        description="Resolution for mesh upscaling/refinement",
        items=[
            ('512', '512³', 'Works on 6GB VRAM - Good quality'),
            ('1024', '1024³', 'Requires 12GB+ VRAM - Best quality'),
        ],
        default='512',  # Default to 512 for safety (works on 6GB)
    )

    texture_size: IntProperty(
        name="Texture Size",
        description="Size of the output texture map",
        default=1024,
        min=512,
        max=4096,
    )
    
    export_mode: EnumProperty(
        name="Export Mode",
        description="Choose whether to export mesh only or mesh with texture",
        items=[
            ('MESH_ONLY', "Mesh Only", "Export only the 3D mesh without texture baking"),
            ('MESH_TEXTURE', "Mesh + Texture", "Export mesh with baked texture"),
        ],
        default='MESH_TEXTURE',
    )

    center_object: BoolProperty(
        name="Center Object",
        description="Center the generated object at the world origin",
        default=True,
    )

    scale_object: FloatProperty(
        name="Scale Object",
        description="Scale factor for the generated object",
        default=1.0,
        min=0.1,
        max=10.0,
    )


class TRELLIS_OT_InstallDependencies(Operator):
    """Install TRELLIS dependencies"""
    bl_idname = "trellis.install_dependencies"
    bl_label = "Install Dependencies"
    bl_description = "Install PyTorch, TRELLIS, and all required dependencies into Blender's Python"
    bl_options = {'REGISTER'}

    use_separate_console: BoolProperty(
        name="Use Separate Console",
        description="Open installation in a separate console window (Windows only)",
        default=True
    )

    def execute(self, context):
        """Execute the operator - tries to launch separate console first"""
        import subprocess
        import platform
        import sys

        # Get preferences
        preferences = context.preferences.addons[__package__].preferences

        # Try separate console on Windows
        if self.use_separate_console and platform.system() == 'Windows':
            try:
                # Get script path
                addon_dir = os.path.dirname(__file__)
                install_script = os.path.join(addon_dir, "install_in_console.py")

                # Verify script exists
                if not os.path.exists(install_script):
                    raise FileNotFoundError(f"Install script not found: {install_script}")

                # Verify Python executable
                if not os.path.exists(sys.executable):
                    raise FileNotFoundError(f"Python executable not found: {sys.executable}")

                print(f"\n{'='*70}")
                print("🚀 Launching installation in separate console...")
                print(f"   Script: {install_script}")
                print(f"   Python: {sys.executable}")
                print(f"   Working dir: {addon_dir}")

                # Use Windows 'start' command with cmd /k to keep console open
                # This keeps the window open even if there's an error
                python_exe = sys.executable
                script_path = install_script

                # Use cmd /k to keep console open
                # Format: start "title" cmd /k "python.exe script.py && pause"
                # The && pause keeps it open after completion
                cmd = f'start "TRELLIS Installation" cmd /k "cd /d "{addon_dir}" && "{python_exe}" "{script_path}" && pause"'

                print(f"   Command: {cmd}")

                # Also show log file location
                import pathlib
                log_file = pathlib.Path.home() / "Documents" / "TRELLIS_install.log"
                print(f"   Log file: {log_file}")
                print(f"   (If console closes immediately, check the log file!)")

                process = subprocess.Popen(
                    cmd,
                    shell=True,  # Required for 'start' command
                    cwd=addon_dir
                )

                print(f"   ✓ Process launched (PID: {process.pid})")
                print(f"{'='*70}")

                self.report({'INFO'}, "Installation console opened! Check the new window.")
                print("\n✅ BLENDER WILL NOT FREEZE - Installation runs in separate window")
                print("   → Watch the NEW CONSOLE WINDOW for progress")
                print("   → You can use Blender normally while it installs")
                print(f"   → Installation log: {log_file}")
                print("   → The console will stay open after completion")
                print("\n   If console closes immediately, check the log file!\n")
                return {'FINISHED'}

            except Exception as e:
                self.report({'ERROR'}, f"Could not open console: {e}")
                print(f"\n❌ Could not open separate console:")
                print(f"   Error: {e}")
                print(f"   Type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                print("\n⚠  FALLING BACK TO INLINE INSTALLATION")
                print("⚠  BLENDER WILL FREEZE FOR 10-30 MINUTES - THIS IS NORMAL!")
                print("   Watch this console for progress...\n")
                import time
                time.sleep(3)  # Give user time to read the warning

        # Try terminal on Linux/Mac
        if self.use_separate_console and platform.system() != 'Windows':
            addon_dir = os.path.dirname(__file__)
            install_script = os.path.join(addon_dir, "install_in_console.py")

            terminals = [
                ['gnome-terminal', '--', sys.executable, install_script],
                ['konsole', '-e', sys.executable, install_script],
                ['xterm', '-e', sys.executable, install_script],
            ]

            for term_cmd in terminals:
                try:
                    subprocess.Popen(term_cmd)
                    self.report({'INFO'}, "Installation terminal opened!")
                    return {'FINISHED'}
                except FileNotFoundError:
                    continue

            print("⚠ Could not open terminal, running inline...")

        # Fallback: Run inline installation
        return self._run_inline_installation(context)

    def _run_inline_installation(self, context):
        """Execute the operator"""
        from . import dependency_installer
        import time

        # Get preferences
        preferences = context.preferences.addons[__package__].preferences

        self.report({'INFO'}, "Starting dependency installation...")

        # Header
        print("\n" + "╔" + "═"*68 + "╗")
        print("║" + " "*18 + "TRELLIS Dependency Installation" + " "*19 + "║")
        print("╚" + "═"*68 + "╝")
        print("\n⚠  This process will take 10-30 minutes depending on internet speed")
        print("⚠  Blender may appear frozen - this is normal!")
        print("⚠  Watch this console for progress updates\n")

        start_time = time.time()

        # Create virtual environment first
        print("━" * 70)
        print("📦 Setting up virtual environment...")
        print("━" * 70)
        venv_path = dependency_installer.get_venv_path()
        print(f"Location: {venv_path}")

        if dependency_installer.check_venv_exists():
            print(f"✓ Virtual environment already exists\n")
        else:
            print(f"Creating new virtual environment using Blender's Python...")
            success, msg = dependency_installer.create_venv()
            print(f"{msg}\n")
            if not success:
                self.report({'ERROR'}, "Failed to create virtual environment")
                return {'CANCELLED'}

        # Check for DLL issues on Windows
        import platform
        if platform.system() == 'Windows':
            # Check for PyTorch DLL issues
            print("━" * 70)
            print("🔍 Checking for PyTorch DLL issues...")
            print("━" * 70)
            if dependency_installer.check_pytorch_dll_issue():
                print("⚠ PyTorch DLL issues detected (c10.dll error)!")
                print("   This requires a complete reinstall of PyTorch...\n")
                success, msg = dependency_installer.fix_pytorch_dll_windows()
                print(f"   {msg}\n")
                if not success:
                    self.report({'WARNING'}, "PyTorch DLL fix failed - trying full installation")
            else:
                print("✓ No PyTorch DLL issues detected\n")

            # Check for NumPy DLL issues
            print("━" * 70)
            print("🔍 Checking for NumPy DLL issues...")
            print("━" * 70)
            if dependency_installer.check_numpy_dll_issue():
                print("⚠ NumPy DLL issues detected!")
                print("   Attempting to fix...\n")
                success, msg = dependency_installer.fix_numpy_dll_windows()
                print(f"   {msg}\n")
                if not success:
                    self.report({'WARNING'}, "NumPy DLL fix failed - installation may have issues")
            else:
                print("✓ No NumPy DLL issues detected\n")

        # Check current status
        print("━" * 70)
        print("📋 Checking current installation status...")
        print("━" * 70)
        status = dependency_installer.get_installation_status()

        torch_installed = status.get('torch', {}).get('installed', False)
        trellis_installed = status.get('trellis', {}).get('installed', False)

        # Check if PyTorch needs installation or replacement
        needs_pytorch_install = False
        if torch_installed:
            torch_version = status.get('torch', {}).get('version', '')
            if '+cpu' in torch_version:
                print(f"\n⚠ Found CPU-only PyTorch {torch_version}")
                print("   Replacing with CUDA version...\n")
                needs_pytorch_install = True

                # Uninstall CPU version first
                print("   Removing CPU-only PyTorch...")
                try:
                    python = dependency_installer.get_python_executable()
                    import subprocess
                    subprocess.run(
                        [python, '-m', 'pip', 'uninstall', 'torch', 'torchvision', '-y'],
                        capture_output=True
                    )
                    print("   ✓ CPU version removed\n")
                except Exception as e:
                    print(f"   ⚠ Warning: Could not uninstall CPU version: {e}\n")
            else:
                print(f"✓ PyTorch already installed: {torch_version}")
        else:
            needs_pytorch_install = True

        if needs_pytorch_install:
            print("\n" + "━" * 70)
            print("🔥 Installing PyTorch 2.x with CUDA 11.8 support...")
            print("━" * 70)
            self.report({'INFO'}, "Installing PyTorch (this may take 5-10 minutes)...")
            print("   📦 Downloading PyTorch (~2GB)...")
            print("   ⏳ This will take 5-10 minutes, please be patient...\n")

            cuda_version = 'cu124'  # Default CUDA 12.4
            success, msg = dependency_installer.install_pytorch(cuda_version)
            print("   " + msg)

            if not success:
                print("\n❌ PyTorch installation failed!")
                self.report({'ERROR'}, "Failed to install PyTorch. Check console for details.")
                return {'CANCELLED'}
            print("   ✅ PyTorch with CUDA installed successfully!")

        # Install TRELLIS dependencies
        print("\n" + "━" * 70)
        print("📦 Installing TRELLIS Dependencies...")
        print("━" * 70)
        self.report({'INFO'}, "Installing TRELLIS dependencies...")

        failed_packages = []
        total_packages = len(dependency_installer.TRELLIS_DEPENDENCIES)
        installed_count = 0

        for i, package in enumerate(dependency_installer.TRELLIS_DEPENDENCIES, 1):
            pkg_name = package.split('>=')[0].split('==')[0]

            # Check if already installed
            installed, version = dependency_installer.check_package_installed(pkg_name)
            if installed:
                print(f"[{i}/{total_packages}] ✓ {pkg_name} ({version}) - already installed")
                installed_count += 1
                continue

            # Install package
            print(f"[{i}/{total_packages}] 📥 Installing {pkg_name}...", end=" ", flush=True)
            success, msg = dependency_installer.install_package(package)

            if success:
                print("✅")
                installed_count += 1
            else:
                print(f"❌")
                print(f"    Error: {msg}")
                failed_packages.append(pkg_name)

        print(f"\n   Installed {installed_count}/{total_packages} packages successfully")
        print("\n✓ TRELLIS package bundled with addon")

        # Install voxelize
        print("\n" + "━" * 70)
        print("📦 Installing Voxelize (for Direct3D-S2)...")
        print("━" * 70)
        success, msg = dependency_installer.install_voxelize()
        if success:
            print(f"✅ {msg}")
        else:
            print(f"❌ {msg}")
            failed_packages.append("voxelize")

        # Verify installation
        print("\n" + "━" * 70)
        print("🔍 Verifying Installation...")
        print("━" * 70)
        success, messages = dependency_installer.verify_installation()

        for msg in messages:
            print(f"   {msg}")

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        print("\n" + "━" * 70)

        if success and not failed_packages:
            # Success message
            print("\n╔" + "═"*68 + "╗")
            print("║" + " "*22 + "Installation Complete!" + " "*23 + "║")
            print("╚" + "═"*68 + "╝")
            print(f"\n✅ All dependencies installed successfully in {minutes}m {seconds}s!")
            print("\n📋 Next Steps:")
            print("   1. Click 'Initialize TRELLIS' button in the addon panel")
            print("   2. Wait for models to download (~5-10 minutes)")
            print("   3. Start generating 3D models!\n")

            self.report({'INFO'}, "✓ Installation completed successfully!")

            # Invalidate status cache to force UI refresh
            from . import ui
            ui.invalidate_status_cache()

            return {'FINISHED'}
        elif failed_packages:
            print("\n╔" + "═"*68 + "╗")
            print("║" + " "*18 + "Installation Completed with Warnings" + " "*14 + "║")
            print("╚" + "═"*68 + "╝")
            print(f"\n⚠  Installation completed in {minutes}m {seconds}s with some warnings")
            print(f"   Failed packages: {', '.join(failed_packages)}")
            print("\n   You can try installing these manually or continue without them.")
            print("   Most functionality should still work.\n")

            self.report({'WARNING'}, f"Installation completed with warnings. Failed: {', '.join(failed_packages)}")
            return {'FINISHED'}
        else:
            print("\n╔" + "═"*68 + "╗")
            print("║" + " "*22 + "Installation Failed!" + " "*24 + "║")
            print("╚" + "═"*68 + "╝")
            print("\n❌ Installation verification failed")
            print("   Please check the error messages above and try again.\n")

            self.report({'ERROR'}, "Installation verification failed. Check console for details.")
            return {'CANCELLED'}


class TRELLIS_OT_InitializePipeline(Operator):
    """Initialize TRELLIS pipeline"""
    bl_idname = "trellis.initialize_pipeline"
    bl_label = "Initialize TRELLIS"
    bl_description = "Load TRELLIS models (required before first use)"
    bl_options = {'REGISTER'}

    use_separate_console: BoolProperty(
        name="Use Separate Console",
        description="Open initialization in a separate console window (Windows only)",
        default=True
    )

    def execute(self, context):
        """Execute the operator - tries to launch separate console first"""
        import subprocess
        import platform
        import sys

        # Get preferences
        preferences = context.preferences.addons[__package__].preferences
        use_cuda = preferences.use_cuda

        # Check if models are already cached - if so, run lightweight inline init
        from . import dependency_installer
        import pathlib

        # Check HuggingFace cache for TRELLIS models
        hf_cache = pathlib.Path.home() / ".cache" / "huggingface" / "hub"
        image_model_cached = (hf_cache / "models--microsoft--TRELLIS-image-large").exists()
        text_model_cached = (hf_cache / "models--microsoft--TRELLIS-text-xlarge").exists()

        models_cached = image_model_cached and text_model_cached

        if models_cached:
            # Models already downloaded, run lightweight inline initialization
            print("\n" + "="*70)
            print("✓ Models already cached - running lightweight initialization")
            print("="*70)
            return self._run_inline_initialization(context)

        # Models not cached yet - use separate console for download
        # Try separate console on Windows
        if self.use_separate_console and platform.system() == 'Windows':
            try:
                # Get script path
                addon_dir = os.path.dirname(__file__)
                init_script = os.path.join(addon_dir, "initialize_in_console.py")

                # Verify script exists
                if not os.path.exists(init_script):
                    raise FileNotFoundError(f"Initialization script not found: {init_script}")

                # Verify Python executable
                if not os.path.exists(sys.executable):
                    raise FileNotFoundError(f"Python executable not found: {sys.executable}")

                print(f"\n{'='*70}")
                print("🚀 Launching initialization in separate console...")
                print(f"   Script: {init_script}")
                print(f"   Python: {sys.executable}")
                print(f"   Working dir: {addon_dir}")

                # Use Windows 'start' command with cmd /k to keep console open
                python_exe = sys.executable
                script_path = init_script

                # Use cmd /k to keep console open
                cmd = f'start "TRELLIS Initialization" cmd /k "cd /d "{addon_dir}" && "{python_exe}" "{script_path}" && pause"'

                print(f"   Command: {cmd}")

                # Also show log file location
                import pathlib
                log_file = pathlib.Path.home() / "Documents" / "TRELLIS_init.log"
                print(f"   Log file: {log_file}")
                print(f"   (If console closes immediately, check the log file!)")

                process = subprocess.Popen(
                    cmd,
                    shell=True,  # Required for 'start' command
                    cwd=addon_dir
                )

                print(f"   ✓ Process launched (PID: {process.pid})")
                print(f"{'='*70}")

                self.report({'INFO'}, "Initialization console opened! Check the new window.")
                print("\n✅ BLENDER WILL NOT FREEZE - Initialization runs in separate window")
                print("   → Watch the NEW CONSOLE WINDOW for progress")
                print("   → You can use Blender normally while it initializes")
                print(f"   → Initialization log: {log_file}")
                print("   → The console will stay open after completion")
                print("\n   If console closes immediately, check the log file!\n")
                return {'FINISHED'}

            except Exception as e:
                self.report({'ERROR'}, f"Could not open console: {e}")
                print(f"\n❌ Could not open separate console:")
                print(f"   Error: {e}")
                print(f"   Type: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                print("\n⚠  FALLING BACK TO INLINE INITIALIZATION")
                print("⚠  BLENDER WILL FREEZE FOR 5-10 MINUTES - THIS IS NORMAL!")
                print("   Watch this console for progress...\n")
                import time
                time.sleep(3)  # Give user time to read the warning

        # Try terminal on Linux/Mac
        if self.use_separate_console and platform.system() != 'Windows':
            addon_dir = os.path.dirname(__file__)
            init_script = os.path.join(addon_dir, "initialize_in_console.py")

            terminals = [
                ['gnome-terminal', '--', sys.executable, init_script],
                ['konsole', '-e', sys.executable, init_script],
                ['xterm', '-e', sys.executable, init_script],
            ]

            for term_cmd in terminals:
                try:
                    subprocess.Popen(term_cmd)
                    self.report({'INFO'}, "Initialization terminal opened!")
                    return {'FINISHED'}
                except FileNotFoundError:
                    continue

            print("⚠ Could not open terminal, running inline...")

        # Fallback: Run inline initialization
        return self._run_inline_initialization(context)

    def _run_inline_initialization(self, context):
        """Execute the operator inline (blocks Blender)"""
        from .pipeline_manager import get_pipeline_manager

        # Get preferences
        preferences = context.preferences.addons[__package__].preferences
        use_cuda = preferences.use_cuda

        # Initialize pipeline
        self.report({'INFO'}, "Initializing TRELLIS... This may take a few minutes.")
        print("\n⚠  BLENDER WILL FREEZE - This is normal!")
        print("⚠  Watch this console for progress\n")

        manager = get_pipeline_manager()

        success = manager.initialize(use_cuda)

        if success:
            self.report({'INFO'}, "TRELLIS initialized successfully!")

            # Invalidate status cache to force UI refresh
            from . import ui
            ui.invalidate_status_cache()

            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Failed to initialize TRELLIS. Check console for details.")
            return {'CANCELLED'}


class TRELLIS_OT_GenerateFromImage(Operator):
    """Generate 3D asset from image"""
    bl_idname = "trellis.generate_from_image"
    bl_label = "Generate from Image"
    bl_description = "Generate 3D asset from input image using TRELLIS"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        """Execute the operator"""
        from .pipeline_manager import get_pipeline_manager
        from .mesh_utils import import_mesh_from_trellis, center_object, scale_object_to_size, convert_tris_to_quads
        props = context.scene.trellis_props
        manager = get_pipeline_manager()
        pipeline_choice = props.pipeline_engine

        if not ensure_pipeline_supported(pipeline_choice, self, mode='TEXT'):
            return {'CANCELLED'}
        pipeline_choice = props.pipeline_engine

        if not ensure_pipeline_supported(pipeline_choice, self, mode='IMAGE'):
            return {'CANCELLED'}

        # Auto-initialize if not initialized
        if not manager.initialized:
            self.report({'INFO'}, "Initializing TRELLIS (first-time setup)...")
            preferences = context.preferences.addons[__package__].preferences
            if not manager.initialize(use_cuda=preferences.use_cuda):
                self.report({'ERROR'}, "Failed to initialize TRELLIS")
                return {'CANCELLED'}
            self.report({'INFO'}, "TRELLIS initialized successfully!")

        # Check input image
        if not props.input_image or not os.path.exists(props.input_image):
            self.report({'ERROR'}, "Invalid input image path")
            return {'CANCELLED'}

        # Determine output formats
        formats = []
        if props.generate_mesh:
            formats.append('mesh')
        if props.generate_gaussian:
            formats.append('gaussian')

        if not formats:
            self.report({'ERROR'}, "No output formats selected")
            return {'CANCELLED'}

        # Use SCENE PROPERTIES (props) for ALL parameters
        seed = props.seed
        sparse_steps = props.sparse_steps
        sparse_cfg = props.sparse_cfg
        slat_steps = props.slat_steps
        slat_cfg = props.slat_cfg
        texture_size = int(props.texture_size)
        # Use 1.0 (no simplification) if checkbox is disabled, otherwise use the slider value
        simplify = props.simplify_mesh if props.enable_simplify_mesh else 1.0

        # Generate (synchronously - blocks Blender but more stable)
        self.report({'INFO'}, f"Generating 3D from image... (seed: {seed})")
        print("\n⚠ BLENDER WILL FREEZE - This is normal during generation!")
        print("⏳ Generation may take 5-15 minutes depending on your GPU")
        print(f"🖼️ Background removal: {'ON' if props.preprocess_image else 'OFF'}\n")

        try:
            outputs = manager.generate_from_image(
                image_path=props.input_image,
                seed=seed,
                sparse_steps=sparse_steps,
                sparse_cfg=sparse_cfg,
                slat_steps=slat_steps,
                slat_cfg=slat_cfg,
                preprocess=props.preprocess_image,
                formats=formats,
                pipeline_key=pipeline_choice,
                simplify_mesh=props.simplify_mesh,

                direct3d_steps=props.direct3d_steps,
                direct3d_guidance=props.direct3d_guidance,
            )

            if outputs is None:
                self.report({'ERROR'}, "Generation failed. Check console for details.")
                return {'CANCELLED'}

            self.report({'INFO'}, "Importing into Blender...")
            obj_name = f"TRELLIS_{os.path.splitext(os.path.basename(props.input_image))[0]}"
            obj = import_mesh_from_trellis(
                outputs,
                name=obj_name,
                texture_size=texture_size,
                simplify=simplify,
                export_mode=props.export_mode
            )

            if obj is None:
                self.report({'ERROR'}, "Failed to import mesh")
                return {'CANCELLED'}

            if props.center_object:
                center_object(obj)
            if props.scale_object:
                scale_object_to_size(obj, target_size=2.0)
            
            if props.convert_to_quads:
                convert_tris_to_quads(obj)

            self.report({'INFO'}, f"Successfully generated and imported {obj_name}!")
            return {'FINISHED'}

        finally:
            manager.cleanup()


class TRELLIS_OT_GenerateFromText(Operator):
    """Generate 3D asset from text"""
    bl_idname = "trellis.generate_from_text"
    bl_label = "Generate from Text"
    bl_description = "Generate 3D asset from text prompt using TRELLIS"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        """Execute the operator"""
        from .pipeline_manager import get_pipeline_manager
        from .mesh_utils import import_mesh_from_trellis, center_object, scale_object_to_size, convert_tris_to_quads

        props = context.scene.trellis_props
        manager = get_pipeline_manager()

        # Auto-initialize if not initialized
        if not manager.initialized:
            self.report({'INFO'}, "Initializing TRELLIS (first-time setup)...")
            preferences = context.preferences.addons[__package__].preferences
            if not manager.initialize(use_cuda=preferences.use_cuda):
                self.report({'ERROR'}, "Failed to initialize TRELLIS")
                return {'CANCELLED'}
            self.report({'INFO'}, "TRELLIS initialized successfully!")

        # Check prompt
        if not props.text_prompt or len(props.text_prompt.strip()) == 0:
            self.report({'ERROR'}, "Please enter a text prompt")
            return {'CANCELLED'}

        # Determine output formats
        formats = []
        if props.generate_mesh:
            formats.append('mesh')
        if props.generate_gaussian:
            formats.append('gaussian')

        if not formats:
            self.report({'ERROR'}, "No output formats selected")
            return {'CANCELLED'}

        # Use SCENE PROPERTIES (props) for generation parameters
        pipeline_choice = props.pipeline_engine
        seed = props.seed
        sparse_steps = props.sparse_steps
        sparse_cfg = props.sparse_cfg
        slat_steps = props.slat_steps
        slat_cfg = props.slat_cfg

        # Generate 3D
        self.report({'INFO'}, f"Generating 3D from text... (seed: {seed})")
        print("\n⚠ BLENDER WILL FREEZE - This is normal during generation!")
        print("⏳ Generation may take 5-15 minutes depending on your GPU\n")

        try:
            outputs = manager.generate_from_text(
                prompt=props.text_prompt,
                seed=seed,
                sparse_steps=sparse_steps,
                sparse_cfg=sparse_cfg,
                slat_steps=slat_steps,
                slat_cfg=slat_cfg,
                formats=formats,
                pipeline_key=pipeline_choice,
            )

            if outputs is None:
                self.report({'ERROR'}, "Generation failed. Check console for details.")
                return {'CANCELLED'}

            # Import into Blender
            self.report({'INFO'}, "Importing into Blender...")

            # Create safe filename from prompt
            safe_name = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in props.text_prompt)
            safe_name = safe_name[:30]  # Limit length
            obj_name = f"TRELLIS_{safe_name}"

            obj = import_mesh_from_trellis(
                outputs,
                name=obj_name,
                texture_size=int(props.texture_size),
                simplify=props.simplify_mesh,
                export_mode=props.export_mode
            )

            if obj is None:
                self.report({'ERROR'}, "Failed to import mesh")
                return {'CANCELLED'}

            # Post-process object
            if props.center_object:
                center_object(obj)

            if props.scale_object:
                scale_object_to_size(obj, target_size=2.0)

            if props.convert_to_quads:
                convert_tris_to_quads(obj)

            self.report({'INFO'}, f"Successfully generated and imported {obj_name}!")
            return {'FINISHED'}

        finally:
            # CRITICAL: Always clean up GPU memory, even on errors!
            manager.cleanup()


class TRELLIS_OT_UpscaleDirect3D(Operator):
    """Upscale existing mesh using Direct3D-S2"""
    bl_idname = "trellis.upscale_direct3d"
    bl_label = "Upscale Mesh"
    bl_description = "Upscale existing mesh using Direct3D-S2 Stage 3"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        """Execute the operator"""
        from .pipeline_manager import get_pipeline_manager
        from .mesh_utils import center_object

        props = context.scene.trellis_props
        manager = get_pipeline_manager()

        # Auto-initialize if not initialized
        if not manager.initialized:
            self.report({'INFO'}, "Initializing TRELLIS (first-time setup)...")
            preferences = context.preferences.addons[__package__].preferences
            if not manager.initialize(use_cuda=preferences.use_cuda):
                self.report({'ERROR'}, "Failed to initialize TRELLIS")
                return {'CANCELLED'}
            self.report({'INFO'}, "TRELLIS initialized successfully!")

        # Check inputs
        if not props.input_image or not os.path.exists(props.input_image):
            self.report({'ERROR'}, "Please select a valid reference image")
            return {'CANCELLED'}
            
        if not props.input_mesh:
            self.report({'ERROR'}, "Please select an input mesh")
            return {'CANCELLED'}

        # Generate 3D
        self.report({'INFO'}, f"Upscaling mesh... (seed: {props.seed})")
        print("\n⚠ BLENDER WILL FREEZE - This is normal during generation!")
        print("⏳ Upscaling may take a few minutes depending on your GPU\n")

        try:
            outputs = manager._upscale_direct3d_mesh(
                image_path=props.input_image,
                mesh_obj=props.input_mesh,
                seed=props.seed,
                simplify_ratio=props.simplify_mesh if props.enable_simplify_mesh else 1.0,
                resolution=int(props.upscale_resolution)  # Pass resolution (512 or 1024)
            )

            if 'error' in outputs:
                self.report({'ERROR'}, f"Upscaling failed: {outputs['error']}")
                return {'CANCELLED'}

            # Import into Blender
            self.report({'INFO'}, "Importing into Blender...")

            # Create safe filename
            obj_name = f"{props.input_mesh.name}_Upscaled"

            # Create mesh from verts/faces
            import bpy
            
            mesh = bpy.data.meshes.new(name=obj_name)
            mesh.from_pydata(outputs['verts'], [], outputs['faces'])
            mesh.update()
            
            obj = bpy.data.objects.new(obj_name, mesh)
            context.collection.objects.link(obj)
            
            # Select new object
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            context.view_layer.objects.active = obj

            # Post-process object
            if props.center_object:
                center_object(obj)

            if props.scale_object != 1.0:
                # Apply scale
                obj.scale = (props.scale_object, props.scale_object, props.scale_object)
                bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

            self.report({'INFO'}, f"Successfully upscaled {obj_name}!")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Error: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
        finally:
            manager.cleanup()


class TRELLIS_OT_UpscaleDirect3DConsole(Operator):
    """Upscale existing mesh in console using Direct3D-S2"""
    bl_idname = "trellis.upscale_direct3d_console"
    bl_label = "Upscale in Console"
    bl_description = "Run mesh upscaling in separate console window with full logging"
    bl_options = {'REGISTER'}

    def execute(self, context):
        """Execute the operator"""
        import subprocess
        import platform
        import sys

        props = context.scene.trellis_props

        # Check inputs
        if not props.input_image or not os.path.exists(props.input_image):
            self.report({'ERROR'}, "Please select a valid reference image")
            return {'CANCELLED'}

        if not props.input_mesh:
            self.report({'ERROR'}, "Please select an input mesh")
            return {'CANCELLED'}

        # Get script path
        addon_dir = os.path.dirname(__file__)
        console_script = os.path.join(addon_dir, "upscale_in_console.py")

        if not os.path.exists(console_script):
            self.report({'ERROR'}, f"Console script not found: {console_script}")
            return {'CANCELLED'}

        # Export mesh to temp file
        import tempfile
        import bpy

        temp_dir = tempfile.gettempdir()
        mesh_path = os.path.join(temp_dir, "upscale_input_mesh.obj")

        # Export selected mesh
        mesh_obj = props.input_mesh

        # Select only this mesh and export
        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj

        try:
            bpy.ops.wm.obj_export(
                filepath=mesh_path,
                export_selected_objects=True,
                export_materials=False,
                export_uv=False
            )
        except Exception as e:
            self.report({'ERROR'}, f"Failed to export mesh: {e}")
            return {'CANCELLED'}

        # Build command parameters
        cmd_params = [
            '--resolution', str(props.upscale_resolution),
            '--seed', str(props.seed),
        ]

        # Add simplification parameters
        if props.enable_simplify_mesh:
            cmd_params.extend(['--simplify', str(props.simplify_mesh)])
        else:
            cmd_params.append('--no-simplify')

        print(f"\n{'='*70}")
        print("🚀 Launching mesh upscaling in separate console...")
        print(f"   Mesh: {mesh_obj.name}")
        print(f"   Reference Image: {props.input_image}")
        print(f"   Resolution: {props.upscale_resolution}³")
        print(f"   Seed: {props.seed}")
        print(f"   Simplify: {'Yes (' + str(props.simplify_mesh) + ')' if props.enable_simplify_mesh else 'No'}")
        print(f"   Script: {console_script}")
        print(f"{'='*70}\n")

        try:
            if platform.system() == 'Windows':
                # Windows: Use proper command list
                cmd = [
                    'cmd.exe', '/k',
                    sys.executable,
                    console_script,
                    mesh_path,
                    props.input_image
                ] + cmd_params

                print(f"   Full command: {' '.join(cmd)}\n")

                # Open new console window
                process = subprocess.Popen(
                    cmd,
                    cwd=addon_dir,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                print(f"   Process launched: PID {process.pid}")
            else:
                # Linux/Mac: Try common terminals
                base_cmd = [sys.executable, console_script, mesh_path, props.input_image] + cmd_params
                terminals = [
                    ['gnome-terminal', '--'] + base_cmd,
                    ['konsole', '-e'] + base_cmd,
                    ['xterm', '-e'] + base_cmd,
                ]
                launched = False
                for term_cmd in terminals:
                    try:
                        subprocess.Popen(term_cmd, cwd=addon_dir)
                        launched = True
                        break
                    except FileNotFoundError:
                        continue

                if not launched:
                    self.report({'ERROR'}, "No terminal emulator found")
                    return {'CANCELLED'}

            self.report({'INFO'}, "Upscale console opened! Check console window for progress.")
            print(f"✓ Console launched successfully")
            print(f"   Output will be saved as: {mesh_path.replace('.obj', '_upscaled.obj')}")
            print(f"   Import the result using File > Import > Wavefront (.obj)\n")

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Failed to launch console: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}





class TRELLIS_OT_GenerateImageConsole(Operator):
    """Generate 3D from image in separate console"""
    bl_idname = "trellis.generate_image_console"
    bl_label = "Generate in Console"
    bl_description = "Run generation in separate console window with full logging"
    bl_options = {'REGISTER'}

    def execute(self, context):
        """Execute the operator"""
        import subprocess
        import platform
        import sys
        import json

        props = context.scene.trellis_props
        pipeline_choice = props.pipeline_engine

        if not ensure_pipeline_supported(pipeline_choice, self, mode='IMAGE'):
            return {'CANCELLED'}

        # Check input image
        if not props.input_image or not os.path.exists(props.input_image):
            self.report({'ERROR'}, "Invalid input image path")
            return {'CANCELLED'}

        # Get script path
        addon_dir = os.path.dirname(__file__)
        console_script = os.path.join(addon_dir, "generate_in_console.py")

        if not os.path.exists(console_script):
            self.report({'ERROR'}, f"Console script not found: {console_script}")
            return {'CANCELLED'}

        # Use SCENE PROPERTIES (props) for ALL parameters
        seed = props.seed

        # Build command with parameters based on pipeline
        generation_params = [
            '--seed', str(seed),
            '--texture-size', str(props.texture_size),
            '--mesh-simplify', str(props.simplify_mesh),
            '--pipeline', pipeline_cli_value(pipeline_choice),
        ]
        
        # Add pipeline-specific parameters
        if pipeline_choice == 'TRELLIS':
            generation_params.extend([
                '--sparse-steps', str(props.sparse_steps),
                '--sparse-cfg', str(props.sparse_cfg),
                '--slat-steps', str(props.slat_steps),
                '--slat-cfg', str(props.slat_cfg),
            ])
        elif pipeline_choice == 'DIRECT3D':
            generation_params.extend([
                '--direct3d-steps', str(props.direct3d_steps),
                '--direct3d-guidance', str(props.direct3d_guidance),
                '--direct3d-resolution', props.direct3d_resolution,
            ])

        # Add export mode
        generation_params.extend(['--export-mode', props.export_mode])

        # Only add texture size if NOT in MESH_ONLY mode
        if props.export_mode != 'MESH_ONLY':
            generation_params.extend(['--texture-size', str(props.texture_size)])


        print(f"\n{'='*70}")
        print("🚀 Launching generation in separate console...")
        print(f"   Image: {props.input_image}")
        print(f"   Pipeline: {pipeline_choice}")
        print(f"   Script: {console_script}")
        if pipeline_choice == 'TRELLIS':
            print(f"   Parameters: seed={seed}, sparse_steps={props.sparse_steps}, slat_steps={props.slat_steps}")
        elif pipeline_choice == 'DIRECT3D':
            print(f"   Parameters: seed={seed}, steps={props.direct3d_steps}, guidance={props.direct3d_guidance}")
        print(f"   Texture: {props.texture_size}x{props.texture_size}, Simplify: {props.simplify_mesh}")
        print(f"{'='*70}\n")

        try:
            if platform.system() == 'Windows':
                # Windows: Use proper command list to avoid argument parsing issues
                cmd = [
                    'cmd.exe', '/k',
                    sys.executable,
                    console_script,
                    props.input_image
                ] + generation_params
                
                print(f"   Full command: {' '.join(cmd)}\n")
                
                # Use creationflags to open new console window
                process = subprocess.Popen(
                    cmd,
                    cwd=addon_dir,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                print(f"   Process launched: PID {process.pid}")
            else:
                # Linux/Mac: Try common terminals
                base_cmd = [sys.executable, console_script, props.input_image] + generation_params
                terminals = [
                    ['gnome-terminal', '--'] + base_cmd,
                    ['konsole', '-e'] + base_cmd,
                    ['xterm', '-e'] + base_cmd,
                ]
                launched = False
                for term_cmd in terminals:
                    try:
                        subprocess.Popen(term_cmd, cwd=addon_dir)
                        launched = True
                        break
                    except FileNotFoundError:
                        continue

                if not launched:
                    self.report({'ERROR'}, "No terminal emulator found")
                    return {'CANCELLED'}

            log_file = os.path.join(os.path.expanduser("~"), "Documents", "TRELLIS_generation.log")
            self.report({'INFO'}, "Generation console opened!")
            print(f"✓ Console launched successfully")
            print(f"📝 Watch the console window for progress")
            print(f"📝 Log file: {log_file}\n")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Failed to launch console: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class TRELLIS_OT_GenerateTextConsole(Operator):
    """Generate 3D from text in separate console"""
    bl_idname = "trellis.generate_text_console"
    bl_label = "Generate in Console"
    bl_description = "Run text-to-3D generation in separate console window with full logging"
    bl_options = {'REGISTER'}

    def execute(self, context):
        """Execute the operator"""
        import subprocess
        import platform
        import sys

        props = context.scene.trellis_props
        pipeline_choice = props.pipeline_engine

        if not ensure_pipeline_supported(pipeline_choice, self, mode='TEXT'):
            return {'CANCELLED'}

        # Check text prompt
        if not props.text_prompt or len(props.text_prompt.strip()) == 0:
            self.report({'ERROR'}, "Please enter a text prompt")
            return {'CANCELLED'}

        # Get script path
        addon_dir = os.path.dirname(__file__)
        console_script = os.path.join(addon_dir, "generate_in_console.py")

        if not os.path.exists(console_script):
            self.report({'ERROR'}, f"Console script not found: {console_script}")
            return {'CANCELLED'}

        # Use SCENE PROPERTIES (props) for ALL parameters
        seed = props.seed

        # Build command with all parameters from props (UI values)
        generation_params = [
            '--text', props.text_prompt,
            '--seed', str(seed),
            '--sparse-steps', str(props.sparse_steps),
            '--sparse-cfg', str(props.sparse_cfg),
            '--slat-steps', str(props.slat_steps),
            '--slat-cfg', str(props.slat_cfg),
            '--slat-cfg', str(props.slat_cfg),
            '--mesh-simplify', str(props.simplify_mesh),
            '--pipeline', pipeline_cli_value(pipeline_choice),
            '--export-mode', props.export_mode,
        ]

        # Only add texture size if NOT in MESH_ONLY mode
        if props.export_mode != 'MESH_ONLY':
            generation_params.extend(['--texture-size', str(props.texture_size)])

        print(f"\n{'='*70}")
        print("🚀 Launching text-to-3D in separate console...")
        print(f"   Prompt: {props.text_prompt}")
        print(f"   Pipeline: {pipeline_choice}")
        print(f"   Script: {console_script}")
        print(f"   Parameters: seed={seed}, sparse_steps={props.sparse_steps}, slat_steps={props.slat_steps}")
        print(f"   Texture: {props.texture_size}x{props.texture_size}, Simplify: {props.simplify_mesh}")
        print(f"{'='*70}\n")

        try:
            if platform.system() == 'Windows':
                # Windows: Build proper command with quoted text prompt
                # The text prompt stays a single argument via list concatenation
                cmd = [
                    'cmd.exe', '/k',
                    sys.executable,
                    console_script,
                ] + generation_params
                
                print(f"   Full command: {' '.join(cmd)}\n")
                
                # Use creationflags to open new console window
                process = subprocess.Popen(
                    cmd,
                    cwd=addon_dir,
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
                print(f"   Process launched: PID {process.pid}")
            else:
                # Linux/Mac: Try common terminals
                base_cmd = [
                    sys.executable,
                    console_script,
                ] + generation_params
                terminals = [
                    ['gnome-terminal', '--'] + base_cmd,
                    ['konsole', '-e'] + base_cmd,
                    ['xterm', '-e'] + base_cmd,
                ]
                launched = False
                for term_cmd in terminals:
                    try:
                        subprocess.Popen(term_cmd, cwd=addon_dir)
                        launched = True
                        break
                    except FileNotFoundError:
                        continue

                if not launched:
                    self.report({'ERROR'}, "No terminal emulator found")
                    return {'CANCELLED'}

            log_file = os.path.join(os.path.expanduser("~"), "Documents", "TRELLIS_generation.log")
            self.report({'INFO'}, "Text-to-3D console opened!")
            print(f"✓ Console launched successfully")
            print(f"📝 Watch the console window for progress")
            print(f"📝 Log file: {log_file}\n")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Failed to launch console: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class TRELLIS_OT_ImportLastGeneration(Operator):
    """Import the last generated GLB file from console generation"""
    bl_idname = "trellis.import_last_generation"
    bl_label = "Import Last Generation"
    bl_description = "Import the most recent GLB file from TRELLIS_Output folder"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        """Execute the operator"""
        from pathlib import Path
        from .mesh_utils import import_glb

        props = context.scene.trellis_props

        # Look for GLB files in output directory
        output_dir = Path.home() / "Documents" / "TRELLIS_Output"
        
        if not output_dir.exists():
            self.report({'ERROR'}, "No TRELLIS_Output folder found")
            return {'CANCELLED'}

        # Find all GLB files
        # Find all generated files (GLB, OBJ, PLY, STL)
        extensions = ['*.glb', '*.obj', '*.ply', '*.stl']
        generated_files = []
        
        for ext in extensions:
            generated_files.extend(list(output_dir.glob(ext)))
        
        if not generated_files:
            self.report({'ERROR'}, "No generated 3D files found in TRELLIS_Output")
            return {'CANCELLED'}

        # Get the most recent file
        latest_file = max(generated_files, key=lambda p: p.stat().st_mtime)
        
        self.report({'INFO'}, f"Importing {latest_file.name}...")
        print(f"\n📦 Importing: {latest_file}")

        # Import based on extension
        suffix = latest_file.suffix.lower()
        obj = None
        
        if suffix == '.glb':
            obj = import_glb(str(latest_file), collection_name="TRELLIS")
        elif suffix == '.obj':
            bpy.ops.wm.obj_import(filepath=str(latest_file))
            obj = context.selected_objects[0] if context.selected_objects else None
        elif suffix == '.ply':
            bpy.ops.wm.ply_import(filepath=str(latest_file))
            obj = context.selected_objects[0] if context.selected_objects else None
        elif suffix == '.stl':
            bpy.ops.wm.stl_import(filepath=str(latest_file))
            obj = context.selected_objects[0] if context.selected_objects else None
            
        if obj:
            # Rename object to match file
            obj.name = latest_file.stem
        
        if obj is None:
            self.report({'ERROR'}, "Failed to import GLB file")
            return {'CANCELLED'}

        # Apply post-processing
        from .mesh_utils import center_object, scale_object_to_size

        if props.center_object:
            center_object(obj)

        if props.scale_object:
            scale_object_to_size(obj, target_size=2.0)

        self.report({'INFO'}, f"Successfully imported {latest_file.name}!")
        return {'FINISHED'}


class TRELLIS_OT_OpenPreferences(Operator):
    """Open addon preferences"""
    bl_idname = "trellis.open_preferences"
    bl_label = "Open Preferences"
    bl_description = "Open TRELLIS addon preferences (includes Smart Offloading checkbox)"
    bl_options = {'REGISTER'}

    def execute(self, context):
        """Execute the operator"""
        # Open preferences and show this addon
        bpy.ops.preferences.addon_show(module=__package__)
        return {'FINISHED'}
